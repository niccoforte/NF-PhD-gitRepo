"""Independent true-field curve-loss attribution; imported training components."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from resources.MLdual import DualTaskTransformer,_json_safe,_move_to_device
from resources.MLfunc import CombinedCurveLoss,curve_default_zone_boundaries
from resources.MLmodels import _model_optimizer,_model_scheduler

TERMS={"zone":("weighted_mse",.5),"derivative":("derivative",.05),
       "peak_height":("peak",.25),"work":("energy",.1),
       "peak_location":("peak_location",.02)}
VARIANTS=("mse","range_mse",*TERMS,"combined")


class CurveTermObjective(nn.Module):
    """Keep normalized MSE fixed while adding exactly one physical curve term.

    range_mse alone isolates the historical per-target-range base replacement.
    combined tests interactions, not attribution of a single constituent.
    """
    def __init__(self,variant,mean,scale,x_values,zone_boundaries=(65,130)):
        super().__init__()
        if variant not in VARIANTS:raise ValueError("Unknown curve-term variant")
        self.variant=variant
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        self.terms=CombinedCurveLoss(mse_weight=1,weighted_mse_weight=.5,derivative_weight=.05,
            peak_weight=.25,energy_weight=.1,peak_location_weight=.02,
            x_values=x_values,zone_boundaries=zone_boundaries,zone_weights=(1,5,.2),
            SoftPeak_beta=20,peak_target_mode="soft")

    def components(self,pred,target):
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Nonfinite independent curve prediction/target")
        mse=(pred-target).square().mean()
        raw={"mse":mse};weighted={"mse":mse}
        p,y=pred*self.scale+self.mean,target*self.scale+self.mean
        if self.variant=="range_mse":
            raw["range_mse"]=self.terms._mse_component(p,y)
            weighted={"range_mse":raw["range_mse"]}
        else:
            for name,(attribute,coefficient) in TERMS.items():
                if self.variant in (name,"combined"):
                    raw[name]=getattr(self.terms,attribute)(p,y)
                    weighted[name]=coefficient*raw[name]
        return sum(weighted.values()),raw,weighted

    def forward(self,pred,target):return self.components(pred,target)[0]


class IndependentCurveTrainer:
    """One task's curve Transformer, optimizer and early stopping; no field model."""
    def __init__(self,stage,data,mode,variant,training,device="cuda"):
        self.device=torch.device(device);self.stage=stage.to(self.device);self.data=data
        self.mode,self.variant,self.training=mode,variant,dict(training)
        stats=data.normalizers["curve"][mode]
        self.objective=CurveTermObjective(variant,**stats,x_values=data.metadata["curve_x_values"][mode],
            zone_boundaries=curve_default_zone_boundaries(mode)).to(self.device)
        self.loaders={s:v for s,v in data.make_dataloaders(batch_size=training["batch"],num_workers=0).items() if s in ("train","val")}
        true_train=data.inverse_curve(mode,data.splits["train"]["curve"][mode]).astype(float)
        true_val=data.inverse_curve(mode,data.splits["val"]["curve"][mode]).astype(float)
        self.train_mean=true_train.mean(0)
        self.selection_denominator=float(np.mean((true_val-self.train_mean)**2))
        if self.selection_denominator<=0:raise ValueError("Degenerate TRAIN-mean curve reference")
        self.history=[];self.best_epoch=None

    def predict_batch(self,batch):
        return self.stage(batch["field"],batch["task_features"],batch["node_mask"])[self.mode]

    def run_loader(self,split,optimizer=None):
        self.stage.train(optimizer is not None);totals={};count=0;preds=[];ids=[]
        with torch.set_grad_enabled(optimizer is not None):
            for values in self.loaders[split]:
                ids.extend(values["sample_id"]);batch=_move_to_device(values,self.device)
                if optimizer is not None:optimizer.zero_grad(set_to_none=True)
                pred=self.predict_batch(batch);target=batch["curve"][self.mode]
                total,raw,weighted=self.objective.components(pred,target)
                if not torch.isfinite(total):raise FloatingPointError("Nonfinite curve objective")
                if optimizer is not None:
                    total.backward()
                    if self.training["grad_clip"] is not None:
                        nn.utils.clip_grad_norm_(self.stage.parameters(),self.training["grad_clip"])
                    optimizer.step()
                n=len(values["sample_id"]);count+=n
                for key,value in dict(loss=total,**{"raw_"+k:v for k,v in raw.items()},**{"weighted_"+k:v for k,v in weighted.items()}).items():
                    totals[key]=totals.get(key,0.)+float(value.detach())*n
                if optimizer is None:preds.append(pred.cpu().numpy())
        return {k:v/count for k,v in totals.items()},(np.concatenate(preds) if preds else None),np.asarray(ids)

    def fit(self,epochs,patience,delta,output,metadata):
        optimizer=_model_optimizer(self.stage.parameters(),self.training["opt"],self.training["lr"])
        scheduler=_model_scheduler(optimizer,self.training["scheduler"])
        best=float("inf");stale=0;best_state=None;start=time.monotonic()
        for epoch in range(1,epochs+1):
            train,_,_=self.run_loader("train",optimizer);val,pred,_=self.run_loader("val")
            physical=self.data.inverse_curve(self.mode,pred)
            truth=self.data.inverse_curve(self.mode,self.data.splits["val"]["curve"][self.mode])
            mse=float(np.mean((physical.astype(float)-truth)**2));score=mse/self.selection_denominator
            if scheduler is not None:scheduler.step(score)
            self.history.append(dict(epoch=epoch,**{"train_"+k:v for k,v in train.items()},
                **{"val_"+k:v for k,v in val.items()},selection_score=score,curve_rmse=np.sqrt(mse),
                learning_rate=optimizer.param_groups[0]["lr"]))
            if score<best-delta:
                best,stale,self.best_epoch=score,0,epoch
                best_state=copy.deepcopy(self.stage.state_dict());self.save(output,metadata)
            else:stale+=1
            pd.DataFrame(self.history).to_csv(Path(output)/"loss_history.csv",index=False)
            print(f"CURVE_TERMS {self.mode} {self.variant} epoch {epoch}: selection={score:.6g}, RMSE={np.sqrt(mse):.6g}",flush=True)
            if stale>=patience:break
        self.stage.load_state_dict(best_state)
        metadata.update(training_seconds=time.monotonic()-start,best_selection_score=best,
            active_parameters=sum(p.numel() for p in self.stage.parameters()))
        self.save(output,metadata)

    def save(self,path,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        config=dict(kind="independent-curve-loss-terms",version=1,model_config=self.stage.get_config(),
            mode=self.mode,variant=self.variant,training=self.training,best_epoch=self.best_epoch,
            term_coefficients=TERMS,selection_denominator=self.selection_denominator,
            data=self.data.to_metadata(),metadata=metadata,run_layout=metadata.get("run_layout"))
        (path/"model.json").write_text(json.dumps(_json_safe(config),indent=2))
        self.data.to_json(path/"model_data.json");torch.save(self.stage.state_dict(),path/"model.mdl")

    def export(self,path,metadata):
        from resources.MLmetrics import curve_performance_diagnostics
        from resources.MLstudy_reports import measure_curve_regions
        _,pred,ids=self.run_loader("val");mode=self.mode;data=self.data;folder=Path(path)/"results"
        folder.mkdir(exist_ok=True)
        p=data.inverse_curve(mode,pred);y=data.inverse_curve(mode,data.splits["val"]["curve"][mode])
        x=data.metadata["curve_x_values"][mode]
        diagnostics=curve_performance_diagnostics(p,y,x_values=x,
            zone_boundaries=curve_default_zone_boundaries(mode),
            train_truth=data.inverse_curve(mode,data.splits["train"]["curve"][mode]))
        for key,table in diagnostics.items():
            if isinstance(table,pd.DataFrame):
                table=table.copy()
                if key=="sample_metrics":table["sample_id"]=ids
                table.to_csv(folder/f"{mode}_val_curve_{key}.csv",index=False)
        pd.DataFrame(measure_curve_regions(p,y,x,self.train_mean)).to_csv(folder/"curve_regions.csv",index=False)
        np.savez(folder/"predictions.npz",sample_ids=ids,**{f"{mode}_val_curve_outputs":p,
            f"{mode}_val_curve_truth":y,f"{mode}_curve_x_values":x})
        metrics=dict(kind="independent-curve-loss-terms",evaluation_split="val",prediction_space="physical",
            prediction_scope=dict(kinds=["curve"],modes=[mode]),best_epoch=self.best_epoch,
            curve_training_source="true",curve_evaluation_source="true",metadata=metadata,
            diagnostics=dict(curve={mode:diagnostics["summary"]}))
        (folder/"metrics.json").write_text(json.dumps(_json_safe(metrics),indent=2))
        return folder


def load_curve_term_model(path,device="cpu"):
    path=Path(path);cfg=json.loads((path/"model.json").read_text())
    if cfg.get("kind")!="independent-curve-loss-terms" or cfg.get("version")!=1:
        raise ValueError("Not an independent curve-term checkpoint")
    stage=DualTaskTransformer(**cfg["model_config"]).to(device)
    stage.load_state_dict(torch.load(path/"model.mdl",map_location=device,weights_only=True));stage.eval()
    return stage,cfg

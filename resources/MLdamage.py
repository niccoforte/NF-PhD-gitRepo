"""STATUS supervision on native struts, sharing the existing nodal field encoder.

No true damage enters inference. DATA/MODEL and old DUAL descriptors are unchanged.
Study entry points and reports live in HPC/studies/Damage, not this library.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, precision_recall_curve

from resources.MLdual import _json_safe, _move_to_device


def align_status_bank(path, data, mode, splits=("train", "val")):
    """Join exact specimen IDs, native endpoints, initial geometry and frame axes."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if str(a["schema_version"]) != "strut_status_bank_v1" or str(a["task"]) != mode or str(a["mech_mode"]) != "MULTI":
            raise ValueError("Require the matched MULTI STATUS bank")
        ids = list(map(str, a["sample_ids"]))
        if len(set(ids)) != len(ids): raise ValueError("Duplicate damage sample IDs")
        lookup = {v: i for i, v in enumerate(ids)}
        mapping = np.asarray(data.metadata["field_to_canonical"][mode], int)
        xy = np.asarray(data.metadata["canonical_coords"])
        if not np.allclose(a["node_coords"], xy[mapping], rtol=0, atol=2e-5):
            raise ValueError("Damage native node order/coordinates differ from field data")
        edges = mapping[a["edge_index"]]
        if len(np.unique(np.sort(edges,axis=1),axis=0))!=len(edges) or np.any(edges[:,0]==edges[:,1]):
            raise ValueError("Duplicate or self-connected native struts")
        if not data.node_masks[mode][edges].all(): raise ValueError("Damage edges include absent nodes")
        frames = np.asarray(a["frame_values"])[1:]
        if not np.array_equal(a["frame_index"],np.arange(len(frames)+1)):
            raise ValueError("STATUS frame indices are not the saved ordered axis")
        expected = np.asarray(data.metadata["field_frame_values"][mode])
        if frames.shape != expected.shape or not np.allclose(frames, expected, rtol=0, atol=1e-6):
            raise ValueError("Damage/field retained frame schedules differ")
        values, known = a["D"], a["valid_mask"]
        if values.shape != known.shape or values.dtype != np.uint8 or known.dtype != bool:
            raise ValueError("Invalid STATUS target schema")
        sample_xy = a["sample_node_coords"]
        payload = {}
        stats = data.normalizers["geometry"]
        for split in splits:
            positions = np.array([lookup[str(v)] for v in data.sample_ids[split]], int)
            initial = xy+data.splits[split]["geometry"]*np.asarray(stats["scale"])+np.asarray(stats["mean"])
            if not np.allclose(sample_xy[positions], initial[:, mapping], rtol=0, atol=3e-5):
                raise ValueError("Sample initial geometry differs between damage and fields")
            target = values[positions, 1:].transpose(0, 2, 1).copy()
            mask = known[positions, 1:].transpose(0, 2, 1).copy()
            if not np.isin(target, [0, 1]).all() or target[~mask].any():
                raise ValueError("Unknown STATUS is not an observed intact label")
            payload[split] = dict(target=target, mask=mask)
        return payload, edges, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            schema="strut_status_bank_v1", frame0_dropped=True, retained_frames=len(frames),
            native_strut_ids=a["strut_id"].tolist(), source_metadata=json.loads(str(a["metadata_json"])))


def class_balance(target, mask, cap=1000.):
    labels = target[mask]
    positives, total = int(labels.sum()), len(labels)
    if positives == 0 or positives == total: raise ValueError("Training needs positive and negative STATUS labels")
    p = positives/total
    ratio = min((1-p)/p, cap)
    return dict(positive=positives, negative=total-positives, prevalence=p,
                pos_weight=ratio, cap=cap, denominator=(1-p)+ratio*p)


class BalancedStatusLoss(nn.Module):
    def __init__(self, balance):
        super().__init__(); self.balance = dict(balance)
        self.register_buffer("positive_weight", torch.tensor(balance["pos_weight"], dtype=torch.float32))

    def forward(self, logits, target, mask):
        if not mask.any(): return logits.sum()*0.
        loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask].float(),
                    pos_weight=self.positive_weight)
        return loss/self.balance["denominator"]


class StatusFieldNetwork(nn.Module):
    """One independent field Transformer; symmetric endpoint readout, no feedback."""
    def __init__(self, stage, mode, edges, coords, geometry_normalizer, variant="weak"):
        super().__init__(); self.stage, self.mode, self.variant = stage, mode, variant
        if variant not in ("control", "damage", "weak", "strong"): raise ValueError("Unknown STATUS arm")
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.long))
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))
        for key in ("mean", "scale"):
            self.register_buffer("geometry_"+key, torch.as_tensor(geometry_normalizer[key], dtype=torch.float32))
        self.length_scale = float(np.ptp(coords, axis=0).max())
        self.strut_head = None
        if variant != "control":
            with torch.random.fork_rng():
                # A separate RNG stream leaves common field initialization/dropout unchanged.
                torch.manual_seed(1042)
                self.strut_head = nn.Sequential(nn.LayerNorm(2*stage.d_model+3),
                    nn.Linear(2*stage.d_model+3, stage.d_model), nn.GELU(),
                    nn.Linear(stage.d_model, stage.out_sizes[mode]//2))
        if variant == "damage":
            for p in self.stage.heads.parameters(): p.requires_grad_(False)

    def forward(self, batch):
        fields, features = self.stage(batch["geometry"], batch["task_features"], batch["node_mask"], return_node_features=True)
        result = dict(field=fields[self.mode])
        if self.strut_head is not None:
            h = features[self.mode]; i, j = self.edges.T
            xy = self.coords+batch["geometry"]*self.geometry_scale+self.geometry_mean
            edge = (xy[:, j]-xy[:, i])/self.length_scale
            # Even orientation encoding makes endpoint reversal exactly invariant.
            xx, yy = edge.unbind(-1); length = edge.norm(dim=-1).clamp_min(1e-8)
            geometry = torch.stack(((xx**2-yy**2)/length**2, 2*xx*yy/length**2, length), -1)
            pair = torch.cat((.5*(h[:, i]+h[:, j]), torch.abs(h[:, i]-h[:, j]), geometry), -1)
            result["damage_logits"] = self.strut_head(pair)
        return result

    def get_config(self):
        return dict(stage=self.stage.get_config(), mode=self.mode, edges=self.edges.cpu().tolist(),
            coords=self.coords.cpu().tolist(), geometry_normalizer={k:getattr(self,"geometry_"+k).cpu().tolist() for k in ("mean","scale")}, variant=self.variant)

    @classmethod
    def from_config(cls, config):
        from resources.MLdual import DualTaskTransformer
        config = dict(config); config["stage"] = DualTaskTransformer(**config["stage"])
        return cls(**config)


class StatusDataset(Dataset):
    def __init__(self, base, status): self.base, self.status = base, status
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        item = self.base[index]
        item.update(damage=torch.as_tensor(self.status["target"][index], dtype=torch.float32),
                    damage_mask=torch.as_tensor(self.status["mask"][index]))
        return item


def status_metrics(probability, target, mask, threshold=.5):
    y, score = target[mask].astype(bool), probability[mask]
    pred = score >= threshold
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
    # Histories stay raw: no cumulative-maximum repair.
    active = (probability >= threshold) & mask
    true_active = target.astype(bool) & mask
    paired = mask[..., 1:] & mask[..., :-1]
    reversals = int((active[..., :-1] & ~active[..., 1:] & paired).sum())
    true_any, pred_any = true_active.any(-1), active.any(-1)
    both = true_any & pred_any
    onset_error = np.abs(active.argmax(-1)-true_active.argmax(-1))
    union = (true_any | pred_any).sum()
    return dict(average_precision=float(average_precision_score(y, score)) if y.any() and (~y).any() else None,
        threshold=float(threshold), precision=precision, recall=recall, f1=2*precision*recall/max(precision+recall,1e-12),
        true_positive=tp, false_positive=fp, missed_fractures=fn, valid_labels=len(y),
        actual_positive_fraction=float(y.mean()), predicted_positive_fraction=float(pred.mean()),
        strut_path_jaccard=float((true_any & pred_any).sum()/union) if union else None,
        onset_interval_mae=float(onset_error[both].mean()) if both.any() else None,
        onset_matched_struts=int(both.sum()), missed_failed_struts=int((true_any & ~pred_any).sum()),
        false_failed_struts=int((~true_any & pred_any).sum()), broken_to_intact_reversals=reversals)


def select_threshold(probability, target, mask):
    p, r, thresholds = precision_recall_curve(target[mask], probability[mask])
    f1 = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


class StatusTrainer:
    """Field-only experimental consumer; separate versioned checkpoint contract."""
    def __init__(self, network, data, status, training, mode, variant, device="cuda"):
        self.network, self.data, self.mode, self.variant = network.to(device), data, mode, variant
        self.device = torch.device(device); self.training = dict(training)
        from resources.MLdualHPO import DualValidationScore
        self.selection_denominator = DualValidationScore(data).baseline_mse["field_"+mode]
        self.balance = class_balance(status["train"]["target"], status["train"]["mask"])
        self.damage_loss = BalancedStatusLoss(self.balance).to(device)
        self.coefficient = dict(control=0., damage=1., weak=.1, strong=1.)[variant]
        self.loaders = {s:DataLoader(StatusDataset(data.datasets[s], status[s]), batch_size=training["batch"],
                                    shuffle=s=="train", num_workers=0) for s in ("train","val")}
        self.history, self.best_epoch = [], None

    def objective(self, outputs, batch):
        mode = self.mode; mask = batch["field_mask"][mode] & batch["node_mask"][mode].unsqueeze(-1)
        residual = outputs["field"][mask]-batch["field"][mode][mask]
        field = residual.square().mean()
        damage = self.damage_loss(outputs["damage_logits"],batch["damage"],batch["damage_mask"]) if "damage_logits" in outputs else field*0.
        total = (field if self.variant != "damage" else field*0.)+self.coefficient*damage
        return total, field, damage

    @torch.no_grad()
    def evaluate(self):
        self.network.eval(); sums = np.zeros(3); fields, logits, labels, masks, ids = [], [], [], [], []
        squared, count = 0., 0
        stats = self.data.normalizers["field"][self.mode]
        for batch in self.loaders["val"]:
            ids.extend(batch["sample_id"]); batch = _move_to_device(batch,self.device)
            out = self.network(batch); total, field, damage = self.objective(out,batch)
            sums += np.array([total.item(),field.item(),damage.item()])*len(batch["sample_id"])
            fields.append(out["field"].cpu().numpy())
            valid = batch["field_mask"][self.mode] & batch["node_mask"][self.mode].unsqueeze(-1)
            scale = torch.as_tensor(stats["scale"],device=self.device,dtype=torch.float32)
            residual = (out["field"]-batch["field"][self.mode])*scale
            sample_counts = valid.flatten(1).sum(-1)
            if not (sample_counts > 0).all(): raise ValueError("Validation specimen has no field targets")
            squared += (residual.double().square().masked_fill(~valid,0).flatten(1).sum(-1)/sample_counts).sum().item()
            count += len(sample_counts)
            if "damage_logits" in out:
                logits.append(out["damage_logits"].sigmoid().cpu().numpy());labels.append(batch["damage"].cpu().numpy()); masks.append(batch["damage_mask"].cpu().numpy())
        result = dict(field=np.concatenate(fields), sample_ids=np.array(ids), field_mse=squared/count,
                      losses=sums/len(self.loaders["val"].dataset))
        if logits:
            result.update(probability=np.concatenate(logits), damage=np.concatenate(labels), damage_mask=np.concatenate(masks))
            result["damage_metrics"] = status_metrics(result["probability"],result["damage"],result["damage_mask"])
        return result

    def fit(self, epochs, patience, delta, output, metadata):
        optimizer = torch.optim.AdamW([p for p in self.network.parameters() if p.requires_grad],
            lr=self.training["lr"],weight_decay=self.training["opt"][1])
        scheduler_config = self.training["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=scheduler_config[2],
            patience=scheduler_config[3],threshold=scheduler_config[4])
        best, best_damage_loss, stale = float("inf"), float("inf"), 0
        start = time.monotonic(); best_state = None
        for epoch in range(1,epochs+1):
            self.network.train(); sums = np.zeros(3)
            for batch in self.loaders["train"]:
                batch = _move_to_device(batch,self.device);optimizer.zero_grad(set_to_none=True)
                total,field,damage = self.objective(self.network(batch),batch)
                if not torch.isfinite(total): raise FloatingPointError("Nonfinite STATUS training loss")
                total.backward()
                if self.training["grad_clip"] is not None: nn.utils.clip_grad_norm_(self.network.parameters(),self.training["grad_clip"])
                optimizer.step();sums += np.array([total.item(),field.item(),damage.item()])*len(batch["sample_id"])
            val = self.evaluate(); train = sums/len(self.loaders["train"].dataset)
            ap = val.get("damage_metrics",{}).get("average_precision")
            if self.variant == "damage" and ap is None: raise ValueError("Validation damage classes unavailable")
            score = 1-ap if self.variant == "damage" else val["field_mse"]/self.selection_denominator
            scheduler.step(score)
            row = dict(epoch=epoch,train_loss=train[0],train_field=train[1],train_damage=train[2],
                val_loss=val["losses"][0],val_field=val["losses"][1],val_damage=val["losses"][2],
                field_rmse=np.sqrt(val["field_mse"]),average_precision=ap,selection_score=score,
                learning_rate=optimizer.param_groups[0]["lr"])
            self.history.append(row)
            tie = self.variant == "damage" and abs(score-best)<=1e-12 and val["losses"][2]<best_damage_loss
            if score < best-delta or tie:
                best,best_damage_loss,stale,self.best_epoch=score,val["losses"][2],0,epoch
                best_state={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
                self.save(output,metadata,best_state)
            else: stale+=1
            print(f"STATUS {self.mode} {self.variant} epoch{epoch}: fieldRMSE={row['field_rmse']:.6g}, AP={ap}, selection={score:.6g}",flush=True)
            if stale>=patience: break
        self.network.load_state_dict(best_state)
        metadata.update(training_seconds=time.monotonic()-start,best_selection_score=best)
        self.save(output,metadata)
        return self.evaluate()

    def save(self, output, metadata, state=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        config=dict(kind="independent-status-field",version=1,model_config=self.network.get_config(),
            training=self.training,balance=self.balance,damage_coefficient=self.coefficient,best_epoch=self.best_epoch,
            selection_denominator=self.selection_denominator,
            history=self.history,metadata=metadata,data=self.data.to_metadata(),run_layout=metadata.get("run_layout"))
        (output/"model.json").write_text(json.dumps(_json_safe(config),indent=2))
        torch.save(self.network.state_dict() if state is None else state,output/"model.mdl")


def load_status_model(path, device="cpu"):
    path=Path(path);config=json.loads((path/"model.json").read_text())
    if config.get("kind")!="independent-status-field" or config.get("version")!=1:
        raise ValueError("Not a STATUS study checkpoint")
    model=StatusFieldNetwork.from_config(config["model_config"]).to(device)
    model.load_state_dict(torch.load(path/"model.mdl",map_location=device,weights_only=True),strict=True)
    model.eval();return model,config

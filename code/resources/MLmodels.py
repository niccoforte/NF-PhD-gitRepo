from resources.imports import *

from resources.MLfunc import train_model, predict_model, plot_loss, plot_StressStrainOUT, absErr
from resources.MLdata import Dataset_

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from torch.serialization import safe_globals
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.preprocessing import MinMaxScaler


class MODEL:
    def __init__(
        self, 
        typ, 
        model, 
        lossf, 
        opt, 
        batch, 
        lr, 
        data, 
        train_dataloader=None, 
        val_dataloader=None, 
        test_dataloader=None, 
        scheduler=None, 
        earlyStop=None, 
        w_init=False, 
        optTrial=None
    ):
        self.typ = typ
        self.model = model
        self.lossf = lossf
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=opt[1])
        self.batch = batch
        self.lr = lr
        self.data = data 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.earlyStop = earlyStop
        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 
                                                                        mode=scheduler[0], 
                                                                        factor=scheduler[1],
                                                                        patience=scheduler[2], 
                                                                        threshold=scheduler[3], 
                                                                        verbose=scheduler[4])
        else:
            self.scheduler = None
        if w_init:
            self.model.apply(w_init)
        self.optTrial = optTrial

        self.trainDS = Dataset_(data.train_in, data.train_out)
        self.valDS = Dataset_(data.val_in, data.val_out)
        self.testDS = Dataset_(data.test_in, data.test_out)

        if train_dataloader is None:
            self.train_dataloader = DataLoader(dataset=self.trainDS, batch_size=self.batch, shuffle=True)
        if val_dataloader is None:
            self.val_dataloader = DataLoader(dataset=self.valDS, batch_size=self.batch, shuffle=True)
        if test_dataloader is None:
            self.test_dataloader = DataLoader(dataset=self.testDS, batch_size=self.batch, shuffle=False)
        
        self.epoch = None
        self.train_lossLog, self.val_lossLog = None, None
        self.best_loss, self.best_rmse, self.best_epoch = None, None, None

        self.test_outputs, self.truth, self.err = None, None, None
        self.best, self.worst = None, None
    
    def train(self, n_epochs, verbose=10, plot=False, RMSEtarget=False):
        self.model, \
            self.epoch, \
            self.train_lossLog, \
            self.val_lossLog, \
            self.best_loss, \
            self.best_rmse, \
            self.best_epoch = train_model(self.typ, 
                                    self.model, 
                                    self.lossf, 
                                    n_epochs, 
                                    self.opt, 
                                    self.train_dataloader, 
                                    val_dataloader=self.val_dataloader, 
                                    scheduler=self.scheduler, 
                                    earlyStop=self.earlyStop, 
                                    verbose=verbose,
                                    optTrial=self.optTrial,
                                    RMSEtarget=RMSEtarget)
        
        if plot:
            plot_loss(self.epoch, self.train_lossLog, self.val_lossLog)
    
    def predict(self, test_dataloader=None, plot=False):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        self.test_outputs, self.truth = predict_model(self.typ,
                                                       self.model, 
                                                       test_dataloader)
        if self.data.scale:
            if "out" in self.data.scale[1].lower() or "all" in self.data.scale[1].lower():
                self.test_outputs = self.data.OUTscaler.inverse_transform(self.test_outputs)
                self.truth = self.data.OUTscaler.inverse_transform(self.truth) 
        if self.data.reduce_dim:
            if "out" in self.data.reduce_dim[1].lower() or "all" in self.data.reduce_dim[1].lower():
                self.test_outputs = self.data.OUTreducer.reconstruct(self.test_outputs)
                self.truth = self.data.OUTreducer.reconstruct(self.truth)

        self.err = absErr(self.test_outputs, self.truth, typ="sum", axis=1)
        self.best, self.worst = self.err.tolist().index(min(self.err)), self.err.tolist().index(max(self.err))
        print(f"Best prediction: {self.best}, Worst prediction: {self.worst}")

        if plot:
            plot_StressStrainOUT(self.data.perOUT_df.T.to_numpy(), self.test_outputs, truth=self.truth, indx=self.best)
            plot_StressStrainOUT(self.data.perOUT_df.T.to_numpy(), self.test_outputs, truth=self.truth, indx=self.worst)
        
    def save(self, path, name):
        torch.save(self.model.state_dict(), f"{path}/{name}.mdl")


### Gaussian Process Regression model
class GPRmodel(GPR):
    def __init__(self, K, restarts, alpha, data=None):
        super(GPRmodel, self).__init__()
        self.K = self.set_K(K)
        self.res = restarts
        self.alpha = alpha
        
        self.GPR = self.build()
        if data:
            self.x = data[0]
            self.y = data[1]
            self.train()
    
    def set_K(self, K):
        return K
    
    def build(self):
        return GPR(kernel=self.K, n_restarts_optimizer=self.res, alpha=self.alpha)
    
    def train(self):
        self.GPR.fit(self.x, self.y)


### Multi-Layer Perceptron model
class resBlock(nn.Module):
    def __init__(self, size, act, norm=None, bias=True):
        super(resBlock, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(size, size*2, bias=bias)
        self.fc2 = nn.Linear(size*2, size, bias=bias)
        self.act = act
        if norm == 'layer':
            self.normL1 = nn.LayerNorm(size*2)
            self.normL2 = nn.LayerNorm(size)
        elif norm == 'batch':
            self.normL1 = nn.BatchNorm1d(size*2)
            self.normL2 = nn.BatchNorm1d(size)

    def forward(self, x):
        skip = x
        x = self.fc1(x)
        if self.norm:
            x = self.normL1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.norm:
            x = self.normL2(x)
        x = x + skip
        x = self.act(x)
        return x #+ skip

class mlpBlock(nn.Module):
    def __init__(self, in_size, out_size, act, norm=None, bias=True):
        super(mlpBlock, self).__init__()
        self.norm = norm
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.act = act
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, act="relu", block="mlp", norm=None, bias=True):
        super(MLP, self).__init__()
        
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

        if len(h_size) > 0:
            self.fcIN = nn.Linear(in_size, h_size[0], bias=bias)
            if block == "mlp":
                self.hlayers = nn.ModuleList([
                    mlpBlock(i, j, self.act, norm, bias) for i, j in zip(h_size[:-1], h_size[1:])])
            elif block == "res":
                self.hlayers = nn.ModuleList([
                    resBlock(i, self.act, norm, bias) for i in h_size])
        else:
            self.hlayers = None
            self.fcIN = None
            h_size = [in_size]
        
        self.fcOUT = nn.Linear(h_size[-1], out_size, bias=bias)

        self.norm = norm
        if norm == 'layer':
            self.normL = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normL = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        

    def forward(self, x):
        if self.fcIN:
            x = self.fcIN(x)
        if self.norm:
            x = self.normL(x)
        if self.fcIN:
            x = self.act(x)
        if self.hlayers:
            for layer in self.hlayers:
                x = layer(x)
        x = self.fcOUT(x)
        return x


### Graph Convolutional Network model  # TODO : Fix GNN predictions.
class gcnBlock(nn.Module):
    def __init__(self, in_size, out_size, act, norm=None):
        super(gcnBlock, self).__init__()
        self.norm = norm
        
        self.Gconv = GCNConv(in_size, out_size)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        if norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x, edge_index):
        x = self.Gconv(x, edge_index)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_size, h_size, out_size, norm=None, pool="mean"):
        super(GCN, self).__init__()
        self.norm = norm
        self.pool = pool
        
        self.GconvIN = GCNConv(in_size, h_size[0])
        # self.GconvOUT = GCNConv(h_size[-1], out_size)
        self.fcOUT = nn.Linear(h_size[-1], out_size)
        self.act = nn.Sigmoid()
        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        self.hlayers = nn.ModuleList([gcnBlock(i, j, self.act, norm) for i, j in zip(h_size[:-1], h_size[1:])])

    def forward(self, x, edge_index, batch):
        x = self.GconvIN(x, edge_index)
        if self.norm: 
            x = self.normIN(x)
        x = self.act(x)
        for layer in self.hlayers:
            x = layer(x, edge_index)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        x = self.fcOUT(x)
        return x

### Graph Attetion Network model
class gatBlock(nn.Module):
    def __init__(self, in_size, out_size, act, heads=1, norm=None):
        super(gatBlock, self).__init__()
        self.norm = norm
        self.heads = heads
        self.act = act
        
        self.GATconv = GATConv(in_size, out_size, heads=heads, concat=True)
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size * heads)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size * heads)
    
    def forward(self, x, edge_index):
        x = self.GATconv(x, edge_index)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class GAT(nn.Module):
    def __init__(self, in_size, h_size, out_size, heads=1, norm=None, pool="mean"):
        super(GAT, self).__init__()
        self.norm = norm
        self.heads = heads
        self.pool = pool.lower()
        self.act = nn.ReLU()

        ## ENCODER
        self.GATconvIN = GATConv(in_size, h_size[0], heads=heads, concat=True)
        self.hlayers = nn.ModuleList([gatBlock(i*heads, j, self.act, heads=heads, norm=norm) 
                                      for i, j in zip(h_size[:-1], h_size[1:])])
        # self.GATconvOUT = GATConv(h_size[-1], out_size // heads, heads=heads)

        ## DECODER
        self.decoder = nn.Sequential(
            nn.Linear(h_size[-1]*heads, 2*h_size[-1]*heads),
            self.act,
            nn.Linear(2*h_size[-1]*heads, out_size),
        )

        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0] * heads)
        elif norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0] * heads)

        self.dropout = nn.Dropout(0.25)
        self.fcOUT = nn.Linear(h_size[-1], out_size)

    def forward(self, x, edge_index, batch):
        x = self.GATconvIN(x, edge_index)
        if self.norm:
            x = self.normIN(x)
        x = self.act(x)
        for layer in self.hlayers:
            x = layer(x, edge_index)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        x = self.decoder(x)
        return x


### Other Models
class Autoencoder(nn.Module):
    def __init__(self, in_size, latent_size, h_size=None, block="mlp"):
        super(Autoencoder, self).__init__()

        self.encoder = MLP(in_size, h_size, latent_size, block=block)
        self.decoder = MLP(latent_size, h_size, in_size, block=block)
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


### Optimization Algorithms
class BoTorchOptimizer:
    def __init__(
        self,
        fitness_fn,
        input_shape,
        DATApath,
        bounds=(-0.2, 0.2),
        X_initial=None,
        y_initial=None,
        resume=False,
        save_log=True,
        save_state=True,
        argv=None
    ):
        self.fitness_fn = fitness_fn
        self.input_shape = input_shape
        self.DATApath = DATApath
        self.dim = np.prod(self.input_shape)
        self.save_log = save_log
        self.save_state = save_state
        self.argv = argv

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        
        log_file = f"{DATApath}/Opt/BO_log.txt"
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.state_file = os.path.join(log_dir, 'BO_optimizer_state.pt')
            
        self.train_X = None
        self.train_Y = None
        self.scaler = None

        self.starting_iteration = 1

        if resume and os.path.exists(self.state_file):
            print(f"--- Resuming optimization from saved state: {self.state_file} ---")
            state = torch.load(self.state_file, weights_only=False)
            self.train_X = state['train_X'].to(self.device, dtype=self.dtype)
            self.train_Y = state['train_Y'].to(self.device, dtype=self.dtype)
            self.scaler = state['scaler']
            self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=self.device, dtype=self.dtype)
            
            for filename in os.listdir(log_dir):
                if filename.startswith("BO_sample") and filename.endswith(".txt"):
                    sample_num = int(filename.split(".")[0][9:])
                    if sample_num >= self.starting_iteration:
                        self.starting_iteration = sample_num + 1
            print(f"Found highest sample number {self.starting_iteration-1} in directory.")
            print(f"Next iteration will be numbered starting from {self.starting_iteration}.")
            
            if self.save_log:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base, ext = os.path.splitext(log_file)
                self.log_file = f"{base}_resume_{timestamp}{ext}"

            print(f"Starting new log file for this run at: {self.log_file}")
        elif X_initial is not None and y_initial is not None:
            print(f"--- Warm-starting optimizer with {len(y_initial)} initial points. ---")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_reshaped = X_initial.reshape(len(y_initial), -1)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped)
            
            self.train_X = torch.tensor(X_scaled, device=self.device, dtype=self.dtype)
            self.train_Y = torch.tensor(y_initial, device=self.device, dtype=self.dtype).unsqueeze(-1)
            self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=self.device, dtype=self.dtype)

            if self.save_log:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base, ext = os.path.splitext(log_file)
                self.log_file = f"{base}_initial_{timestamp}{ext}"
                print(f"Starting new log file for this run at: {self.log_file}")
        else:
            self.bounds = torch.tensor([[bounds[0]] * self.dim, [bounds[1]] * self.dim], device=self.device, dtype=self.dtype)

    def _save_state(self):
        if not self.save_state:
            return
        
        state = {
            'train_X': self.train_X,
            'train_Y': self.train_Y,
            'scaler': self.scaler
        }
        torch.save(state, self.state_file)
        print(f"-> Optimizer state saved to {self.state_file}")
    
    def _get_fitted_model(self):
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _report_progress(self, iteration, history):
        if self.save_log:
            with open(self.log_file, 'a') as f:
                f.write(f"Iteration: {iteration:4} | Best Fitness Found: {max(history):10.4f}\n")
        
        plt.clf()
        plt.plot(history, marker='o', linestyle='-', markersize=4)
        plt.title(f'BoTorch Convergence (Iter: {iteration})')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Score Found')
        plt.grid(True)
        plt.tight_layout()
        plt.pause(0.01)

    def run(self, n_iterations=50, n_initial_points=10, beta=4.0):
        if self.train_X is None:
            print(f"--- Starting fresh run. Generating {n_initial_points} points using LHS. ---")
            sampler = scipy.stats.qmc.LatinHypercube(d=self.dim)
            X_init_np = sampler.random(n=n_initial_points)
            if self.scaler is None:
                low, high = self.bounds[0,0].item(), self.bounds[1,0].item()
                X_init_np = low + (high - low) * X_init_np
            
            y_init_np = np.array([self.fitness_fn(x.reshape(self.input_shape), 0) for x in X_init_np])
            self.train_X = torch.tensor(X_init_np, device=self.device, dtype=self.dtype)
            self.train_Y = torch.tensor(y_init_np, device=self.device, dtype=self.dtype).unsqueeze(-1)
        
        history = [self.train_Y.max().item()]
        self._report_progress(self.starting_iteration - 1, history)

        for i in range(self.starting_iteration, self.starting_iteration + n_iterations):
            print(f"\n--- Iteration {i}/{self.starting_iteration + n_iterations - 1} ---")

            print("1. Fitting the GPR model...")
            model = self._get_fitted_model()
            
            print("2. Redefining and optimizing the acquisition function...")
            acq_function = UpperConfidenceBound(model, beta=beta)
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=20,
                raw_samples=512
            )
            
            new_x_scaled = candidate.detach()
            
            print("3. Evaluating new candidate with fitness function...")
            physical_x_flat = new_x_scaled.cpu().numpy().reshape(1, -1)
            if self.scaler:
                physical_x_flat = self.scaler.inverse_transform(physical_x_flat)
            physical_x = physical_x_flat.reshape(self.input_shape)
            new_y_val = self.fitness_fn(physical_x, i, self.DATApath, self.argv)
            print(f"   -> Score found: {new_y_val:.4f}")
            
            new_y = torch.tensor([[new_y_val]], device=self.device, dtype=self.dtype)
            
            self.train_X = torch.cat([self.train_X, new_x_scaled])
            self.train_Y = torch.cat([self.train_Y, new_y])
            
            history.append(self.train_Y.max().item())
            self._report_progress(i, history)

            self._save_state()

        plt.ioff()
        print("\nOptimization finished.")
        plt.show()
        
        best_idx = self.train_Y.argmax()
        best_solution_scaled = self.train_X[best_idx]
        best_score = self.train_Y[best_idx].item()

        best_solution_physical = best_solution_scaled.reshape(1, -1)
        if self.scaler:
            best_solution_physical = self.scaler.inverse_transform(best_solution_physical)
        return best_solution_physical.reshape(self.input_shape), best_score, self.train_X, self.train_Y


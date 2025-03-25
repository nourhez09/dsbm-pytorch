import torch 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import copy
from omegaconf import OmegaConf
import os
import json
import sys
from datetime import datetime
from DSBM_Gaussian import ScoreNetwork
from tqdm import tqdm  # Make sure tqdm is imported for progress
import wandb  # NEW: Import wandb

# Initialize wandb with initial config.
# When running as a sweep, these values are provided by wandb.
wandb.init(project="dsbm-experiment_2", config={
    "batch_size": 128,
    "lr": 1e-4,
    "num_steps": 20,         # number of steps in the SDE process
    "dataset_size": 5000,    # default dataset size (will be overwritten by sweep)
    "dimension": 2,          # default dimension (will be overwritten by sweep)
    "seed": 42,
    "epsilon": 0.05,         # tolerance for early stopping
    "sigma": 1,              # sigma parameter in the model
    "outer_iters": 1000,     # maximum outer iterations
    "inner_iters": 10000     # inner loop iterations per update
})

# Global hyperparameters (you can adjust as needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = wandb.config.batch_size
lr = wandb.config.lr

# --------------------------
# Helper Function: Compute KL divergence between two Gaussians (from samples)
def compute_gaussian_kl(x, y):
    """
    Approximates the KL divergence between two distributions by assuming that 
    the samples x and y come from diagonal Gaussian distributions.
    x, y: Tensors of shape (N, dim)
    Returns: scalar KL divergence.
    """
    eps = 1e-8  # small constant for numerical stability
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    var_x = x.var(dim=0, unbiased=False)
    var_y = y.var(dim=0, unbiased=False)
    kl = torch.log((var_y + eps).sqrt() / (var_x + eps).sqrt()) \
         + (var_x + (mu_x - mu_y)**2) / (2 * (var_y + eps)) - 0.5
    return kl.sum()

# --------------------------
# The DSBM class remains unchanged.
class DSBM(torch.nn.Module):
    def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
        super().__init__()
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd
        self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        self.first_coupling = first_coupling
    
    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
        z0, z1 = x_pairs[:, 0].to(device), x_pairs[:, 1].to(device)
        t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
        z_t = t * z1 + (1.-t) * z0
        z = torch.randn_like(z_t, device=device)
        z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
        if fb == 'f':
            target = z1 - z0 
            target = target - self.sig * torch.sqrt(t/(1.-t)) * z
        else:
            target = - (z1 - z0)
            target = target - self.sig * torch.sqrt((1.-t)/t) * z
        return z_t, t, target
    
    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
        assert fb in ['f', 'b']
        if prev_model is None:
            assert first_it
            assert fb == 'b'
            zstart = x_pairs[:, 0]
            if self.first_coupling == "ref":
                zend = zstart + torch.randn_like(zstart, device=device) * self.sig
            elif self.first_coupling == "ind":
                zend = x_pairs[:, 1].clone().to(device)
                zend = zend[torch.randperm(len(zend))]
            else:
                raise NotImplementedError
        else:
            assert not first_it
            if prev_model.fb == 'f':
                zstart = x_pairs[:, 0].to(device)
            else:
                zstart = x_pairs[:, 1].to(device)
            zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
        
        if prev_model is not None and prev_model.fb == 'f':
            z0, z1 = zstart, zend
        else:
            z0, z1 = zend, zstart
        return z0, z1
    
    @torch.no_grad()
    def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
        assert fb in ['f', 'b']
        if N is None:
            N = self.N   
        dt = 1./N
        traj = []
        z = zstart.detach().clone().to(device)
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        ts = np.arange(N) / N
        if fb == 'b':
            ts = 1 - ts
        for i in range(N):
            t = torch.ones((batchsize,1), device=device) * ts[i]
            pred = self.net_dict[fb](z, t)
            z = z.detach().clone() + pred * dt
            z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
            traj.append(z.detach().clone())
        return traj

# --------------------------
def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    dsbm_ipf.fb = fb
    optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)
    loss_curve = []
    
    dl = iter(torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
        batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
    
    for i in tqdm(range(inner_iters)):
        try:
            z0, z1 = next(dl)
        except StopIteration:
            dl = iter(torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
            z0, z1 = next(dl)
        
        z_pairs = torch.stack([z0.to(device), z1.to(device)], dim=1)
        z_t, t, target = dsbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
        optimizer.zero_grad()
        pred = dsbm_ipf.net_dict[fb](z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1).mean()
        loss.backward()
        if torch.isnan(loss).any():
            raise ValueError("Loss is nan")
        optimizer.step()
        loss_curve.append(np.log(loss.item()))
        
        # Log the current inner iteration loss for the given direction
        wandb.log({f"Inner_Loss_{fb}": loss.item(), "inner_iteration": i})
    
    return dsbm_ipf, loss_curve

# --------------------------
# Modified run_experiment function that uses hyperparameters from wandb.config
def run_experiment():
    config = wandb.config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    cfg = OmegaConf.create({
        "seed": config.seed,
        "a": 3.0,
        "dim": config.dimension,
        "outer_iters": config.outer_iters,
        "inner_iters": config.inner_iters,
        "net_name": "mlp_small",
        "activation_fn": "torch.nn.Tanh",
        "model_name": "dsbm",
        "fb_sequence": ['b', 'f'],  # alternating backward and forward training
        "num_steps": config.num_steps,
        "sigma": config.sigma,
        "first_coupling": "ref",
        "dataset_size": config.dataset_size
    })
    
    # Update wandb config with experiment-specific parameters (if needed)
    wandb.config.update({
        "seed": config.seed,
        "dimension": config.dimension,
        "dataset_size": config.dataset_size,
        "num_steps": cfg.num_steps,
        "epsilon": config.epsilon,
        "sigma": cfg.sigma
    }, allow_val_change=True)
    
    a = cfg.a
    dim = cfg.dim
    initial_model = torch.distributions.Normal(-a * torch.ones((dim,)), 1)
    target_model = torch.distributions.Normal(a * torch.ones((dim,)), 1)
    
    x0 = initial_model.sample([cfg.dataset_size])
    x1 = target_model.sample([cfg.dataset_size])
    x_pairs = torch.stack([x0, x1], dim=1).to(device)
    
    test_dataset_size = 10000
    x0_test = initial_model.sample([test_dataset_size]).to(device)
    x1_test = target_model.sample([test_dataset_size]).to(device)
    # For backward process, we want to map x1_test back to x0_test, and vice versa.
    x_test_dict = {'f': x0_test, 'b': x1_test}
    
    activation_fn = torch.nn.Tanh
    if cfg.net_name == "mlp_small":
        net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=activation_fn())
    else:
        net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=activation_fn())
    
    model = DSBM(net_fwd=net_fn().to(device), 
                 net_bwd=net_fn().to(device), 
                 num_steps=cfg.num_steps, 
                 sig=cfg.sigma, 
                 first_coupling=cfg.first_coupling)
    train_fn = train_dsbm
    
    model_list = []
    error_record = {
        "mean_error": [],
        "var_error": [],
        "global_loss": [],
        "forward_loss": [],
        "backward_loss": [],
        "kl_rec_markov": [],
        "kl_markov": [],
        "kl_reciprocal": []
    }
    it = 1
    
    # Initialize storage for consecutive KL computations.
    first_forward_state = None
    first_backward_state = None
    last_forward_rec = None
    last_backward_rec = None
    
    while it <= cfg.outer_iters:
        for fb in cfg.fb_sequence:
            first_it = (it == 1)
            if first_it:
                prev_model = None
            else:
                prev_model = model_list[-1]["model"].eval()
            
            # Train one inner loop update in the current direction.
            model, loss_curve = train_fn(model, x_pairs, batch_size, cfg.inner_iters, 
                                         prev_model=prev_model, fb=fb, first_it=first_it)
            model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
            
            # --- Evaluation of Markovian Projection ---
            test_input = x_test_dict['b'] if fb == 'b' else x_test_dict['f']
            traj = model_list[-1]['model'].sample_sde(zstart=test_input, fb=fb)
            initial_state = traj[0]  # starting marginal of the trajectory
            final_state = traj[-1]   # arrival marginal of the trajectory
            
            # --- Evaluation of Reciprocal Projection ---
            recip_z0, recip_z1 = model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)
            rec_sample = recip_z0 if fb == 'b' else recip_z1
            
            # (1) KL between reciprocal projection and Markovian projection:
            kl_rec_markov = compute_gaussian_kl(rec_sample, final_state).item()
            
            # (2) KL between two consecutive Markovian projections:
            if fb == 'b':
                if first_forward_state is not None:
                    kl_markov = compute_gaussian_kl(first_forward_state, final_state).item()
                else:
                    kl_markov = float('nan')
                if first_backward_state is None:
                    first_backward_state = initial_state
            else:  # fb == 'f'
                if first_backward_state is not None:
                    kl_markov = compute_gaussian_kl(first_backward_state, final_state).item()
                else:
                    kl_markov = float('nan')
                if first_forward_state is None:
                    first_forward_state = initial_state
            
            # (3) KL between two consecutive reciprocal projections:   
            if fb == 'b':
                if last_backward_rec is not None:
                    kl_reciprocal = compute_gaussian_kl(last_backward_rec, rec_sample).item()
                else:
                    kl_reciprocal = float('nan')
                last_backward_rec = rec_sample
            else:  # fb == 'f'
                if last_forward_rec is not None:
                    kl_reciprocal = compute_gaussian_kl(last_forward_rec, rec_sample).item()
                else:
                    kl_reciprocal = float('nan')
                last_forward_rec = rec_sample
            
            # Compute errors for the generated distribution.
            learned_mean = final_state.mean(dim=0)
            learned_var  = final_state.var(dim=0)
            mean_error = torch.abs(learned_mean - (-a if fb=='b' else a)).mean().item()
            var_error  = torch.abs(learned_var - 1).mean().item()
            aggregated_error = (mean_error + var_error) / 2.0
            
            forward_loss = loss_curve[0] if fb == 'f' else None
            backward_loss = loss_curve[0] if fb == 'b' else None
            global_loss = loss_curve[0]
            
            error_record["mean_error"].append(mean_error)
            error_record["var_error"].append(var_error)
            error_record["global_loss"].append(global_loss)
            error_record["forward_loss"].append(forward_loss if fb == 'f' else np.nan)
            error_record["backward_loss"].append(backward_loss if fb == 'b' else np.nan)
            error_record["kl_rec_markov"].append(kl_rec_markov)
            error_record["kl_markov"].append(kl_markov)
            error_record["kl_reciprocal"].append(kl_reciprocal)
            
            # Log metrics for this outer iteration
            wandb.log({
                "iteration": it,
                "direction": fb,
                "mean_error": mean_error,
                "var_error": var_error,
                "aggregated_error": aggregated_error,
                "global_loss": global_loss,
                "forward_loss": forward_loss if fb == 'f' else np.nan,
                "backward_loss": backward_loss if fb == 'b' else np.nan,
                "kl_rec_markov": kl_rec_markov,
                "kl_markov": kl_markov,
                "kl_reciprocal": kl_reciprocal,
                "dataset_size": cfg.dataset_size,
                "dimension": cfg.dim,
                "epsilon": config.epsilon,
                "sigma": cfg.sigma
            })
            
            print(f"Iteration {it} [{fb}]: Mean error = {mean_error:.4f}, Variance error = {var_error:.4f}, Aggregated error = {aggregated_error:.4f}")
            print(f"  KL(reciprocal vs Markovian) = {kl_rec_markov:.4f}, KL(Markovian conv.) = {kl_markov:.4f}, KL(Reciprocal conv.) = {kl_reciprocal:.4f}")
            
            if aggregated_error < config.epsilon:
                print(f"Early stopping triggered at iteration {it}: Aggregated error {aggregated_error:.4f} < epsilon {config.epsilon:.4f}")
                return error_record
            
            it += 1
            if it > cfg.outer_iters:
                break
                
    return error_record

def plot_error_evolution(error_record, save_dir="plots_kl_wandb"):
    iterations = np.arange(1, len(error_record["mean_error"]) + 1)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(iterations, error_record["mean_error"], linestyle='-', marker='o', label="Mean Error")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Mean Error")
    plt.title("Mean Error")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(iterations, error_record["var_error"], marker='o', linestyle='-', color='red', label="Variance Error")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Variance Error")
    plt.title("Variance Error")
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(iterations, error_record["global_loss"], marker='o', linestyle='-', label="Global Loss")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Global Loss")
    plt.title("Global Loss")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(iterations, error_record["kl_rec_markov"], marker='o', linestyle='-', label="KL (Reciprocal vs Markovian)")
    plt.xlabel("Outer Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL: Reciprocal vs Markovian")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(iterations, error_record["kl_markov"], marker='o', linestyle='-', label="KL (Consecutive Markovian)")
    plt.xlabel("Outer Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL: Consecutive Markovian")
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(iterations, error_record["kl_reciprocal"], marker='o', linestyle='-', label="KL (Consecutive Reciprocal)")
    plt.xlabel("Outer Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL: Consecutive Reciprocal")
    plt.legend()
    
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"error_evolution_outer{wandb.config.outer_iters}_ds{wandb.config.dataset_size}_dim{wandb.config.dimension}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")
    
    # Log the plot image to wandb
    wandb.log({"Error_Evolution_Plot": wandb.Image(save_path),
               "dataset_size": wandb.config.dataset_size,
               "dimension": wandb.config.dimension,
               "outer_iters": wandb.config.outer_iters})

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()

# New sweep_experiment function: each sweep run executes one experiment.
def sweep_experiment():
    error_record = run_experiment()
    plot_error_evolution(error_record)
    # Optionally, you could save the results to file if needed.
    return error_record

if __name__ == "__main__":
    sweep_experiment()

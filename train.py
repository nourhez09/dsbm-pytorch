import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import copy
from omegaconf import OmegaConf
from DSBM_Gaussian import DSBM, ScoreNetwork, train_dsbm
import os
import json
import sys
from datetime import datetime

# Global hyperparameters (you can adjust as needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# improvement try configuring on cuda to make this faster
batch_size = 128
lr = 1e-4

def run_experiment(outer_iters, dataset_size_param, dimension, inner_iters=10000, seed=42, epsilon=0.01):
    """
    Runs one experiment with given outer iterations, dataset size, and dimension.
    After each outer iteration, the aggregated error (average of mean and variance errors)
    is computed. Training stops early if the aggregated error falls below epsilon.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a configuration using OmegaConf (similar to your hydra config)
    cfg = OmegaConf.create({
        "seed": seed,
        "a": 3.0,               # true parameter a (the two distributions are centered at -a and a)
        "dim": dimension,
        "outer_iters": outer_iters,
        "inner_iters": inner_iters,
        "net_name": "mlp_small",  # or "mlp_large" if desired 
        "activation_fn": "torch.nn.Tanh",  # this string is used in your code to get the class
        "model_name": "dsbm",
        "fb_sequence": ['b', 'f'],  # alternating forward/backward training
        "num_steps": 20, #change to 20
        "sigma": 1,           # sigma parameter (adjust if needed) : noise of EM 
        "first_coupling": "ref",
        "dataset_size": dataset_size_param
    })
    
    # Override the global dataset_size (used when generating the data)
    global dataset_size
    dataset_size = dataset_size_param
    
    # --- Data Generation ---
    a = cfg.a
    dim = cfg.dim
    # Create the initial and target distributions (Normal with mean -a and a, variance=1)
    initial_model = torch.distributions.Normal(-a * torch.ones((dim,)), 1)
    target_model = torch.distributions.Normal(a * torch.ones((dim,)), 1)  
    
    x0 = initial_model.sample([cfg.dataset_size])
    x1 = target_model.sample([cfg.dataset_size])
    x_pairs = torch.stack([x0, x1], dim=1).to(device)
    
    # Also create a fixed test set to evaluate the learned model.
    test_dataset_size = 10000
    x0_test = initial_model.sample([test_dataset_size]).to(device)
    x1_test = target_model.sample([test_dataset_size]).to(device)
    # Here we use the backward direction ('b') so that the generated distribution should match x0.
    x_test_dict = {'f': x0_test, 'b': x1_test}
    
    # --- Network and Model Setup ---
    # For simplicity, we use torch.nn.Tanh directly.
    activation_fn = torch.nn.Tanh
    if cfg.net_name == "mlp_small":
        net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=activation_fn())
    else:
        net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=activation_fn())
    
    # Instantiate the DSBM model using the provided network.
    model = DSBM(net_fwd=net_fn().to(device), 
                 net_bwd=net_fn().to(device), 
                 num_steps=cfg.num_steps, 
                 sig=cfg.sigma, 
                 first_coupling=cfg.first_coupling)
    train_fn = train_dsbm  # use the provided training function for dsbm
    
    # --- Training Loop with Early Stopping and Error Recording ---
    model_list = []
    error_record = {"mean_error": [], "var_error": [], "global_loss": [], "forward_loss": [], "backward_loss": []}
    it = 1
    
    while it <= cfg.outer_iters:
        for fb in cfg.fb_sequence:
            first_it = (it == 1)
            if first_it:
                prev_model = None
            else:
                prev_model = model_list[-1]["model"].eval()
            
            # Train one “inner loop” update using the current direction (fb)
            model, loss_curve = train_fn(model, x_pairs, batch_size, cfg.inner_iters, 
                                         prev_model=prev_model, fb=fb, first_it=first_it)
            # Save a copy of the current model (set to evaluation mode)
            model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
            
            # --- Evaluation: sample using sample_sde ---
            traj = model_list[-1]['model'].sample_sde(zstart=x_test_dict['b'], fb='b')
            final_state = traj[-1]  # final generated distribution
            
            # Compute errors:
            learned_mean = final_state.mean(dim=0)  # mean over samples for each dimension
            learned_var  = final_state.var(dim=0)
            mean_error = torch.abs(learned_mean - (-a)).mean().item()
            var_error  = torch.abs(learned_var - 1).mean().item()
            
            # Aggregate the errors (here we use the average of the two errors)
            aggregated_error = (mean_error + var_error) / 2.0
            
            # Track the losses for forward and backward directions
            forward_loss = loss_curve[0] if fb == 'f' else None
            backward_loss = loss_curve[0] if fb == 'b' else None
            global_loss = loss_curve[0]  # Assuming loss_curve[0] is the global loss for the iteration
            
            error_record["mean_error"].append(mean_error)
            error_record["var_error"].append(var_error)
            error_record["global_loss"].append(global_loss)
            error_record["forward_loss"].append(forward_loss if fb == 'f' else np.nan)
            error_record["backward_loss"].append(backward_loss if fb == 'b' else np.nan)
            
            print(f"Iteration {it}: Mean error = {mean_error:.4f}, Variance error = {var_error:.4f}, Aggregated error = {aggregated_error:.4f}")
            
            # Early stopping check: if aggregated error falls below epsilon, stop early.
            if aggregated_error < epsilon:
                print(f"Early stopping triggered at iteration {it}: Aggregated error {aggregated_error:.4f} < epsilon {epsilon:.4f}")
                return error_record
            
            it += 1
            if it > cfg.outer_iters:
                break
                
    return error_record

def plot_error_evolution(error_record, outer_iters, dataset_size_param, dimension, save_dir="plots"):
    """
    Plots the evolution of the mean, variance, and global errors over outer iterations and saves the plot to a file.
    """
    iterations = np.arange(1, len(error_record["mean_error"]) + 1)
    plt.figure(figsize=(12, 8))
    
    # Plot mean error evolution
    plt.subplot(2, 2, 1)
    plt.plot(iterations, error_record["mean_error"],linestyle='-', marker='o', label="Mean Error")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Error in Mean")
    plt.title(f"Mean Error (dataset size: {dataset_size_param}, dim: {dimension})")
    plt.legend()
    
    # Plot variance error evolution
    plt.subplot(2, 2, 2)
    plt.plot(iterations, error_record["var_error"], marker='o', linestyle='-',color='red', label="Variance Error")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Error in Variance")
    plt.title(f"Variance Error (dataset size: {dataset_size_param}, dim: {dimension})")
    plt.legend()
    
    # Plot global loss evolution
    plt.subplot(2, 2, 3)
    plt.plot(iterations, error_record["global_loss"], marker='o',linestyle='-', label="Global Loss")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Global Loss")
    plt.title("Global Loss")
    plt.legend()
    
    # Plot forward/backward loss evolution
    plt.subplot(2, 2, 4)
    plt.plot(iterations, error_record["forward_loss"], marker='o', linestyle='-',label="Forward Loss")
    plt.plot(iterations, error_record["backward_loss"], marker='o',linestyle='-', label="Backward Loss", color='purple')
    plt.xlabel("Outer Iteration")
    plt.ylabel("Loss")
    plt.title("Forward & Backward Losses")
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a filename for the plot
    filename = f"error_evolution_outer{outer_iters}_ds{dataset_size_param}_dim{dimension}.png"
    save_path = os.path.join(save_dir, filename)

    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory

    print(f"Saved plot to {save_path}")



class Logger:
    """Custom logger to capture print statements."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()


def run_all_experiments(epsilon=0.01, results_filename="experiment_results.json", log_filename="experiment_logs.json"):
    """
    Loops over different values of outer iterations, dataset sizes, and dimensions.
    Saves experiment results to a JSON file and logs terminal output.
    """
    # Define the grid (adjust these values as needed)
    outer_iters_list = [1000]  # maximum number of outer iterations
    dataset_sizes = [5000, 10000]  # different dataset sizes 1000, 2000, 
    dimensions = [2, 5, 10]  # different dimensions 50
    
    experiment_results = {}
    logs = []  # List to store log messages
    
    original_stdout = sys.stdout  # Save original stdout
    sys.stdout = Logger(log_filename)  # Redirect stdout to log file
    
    try:
        for outer_iters in outer_iters_list:
            for ds in dataset_sizes:
                for dim in dimensions:
                    log_message = f"\nRunning experiment: outer_iters={outer_iters}, dataset_size={ds}, dimension={dim}"
                    print(log_message)
                    logs.append({"timestamp": datetime.now().isoformat(), "message": log_message})
                    
                    error_record = run_experiment(outer_iters, ds, dim, epsilon=epsilon)
                    key = f"outer{outer_iters}_ds{ds}_dim{dim}"
                    experiment_results[key] = error_record
                    
                    # Plot error evolution for this experiment
                    plot_error_evolution(error_record, outer_iters,ds, dim)
                    
        # Save experiment results to a JSON file
        with open(results_filename, "w") as f:
            json.dump(experiment_results, f, indent=4)
        
        # Save logs to a JSON file
        with open(log_filename, "w") as f:
            json.dump(logs, f, indent=4)
    
    finally:
        sys.stdout = original_stdout  # Restore original stdout
    
    return experiment_results

# def run_all_experiments(epsilon=0.01):
#     """
#     Loops over different values of outer iterations, dataset sizes, and dimensions.
#     For each combination, runs the experiment with early stopping based on epsilon and plots the error evolution.
#     """
#     # Define the grid (adjust these values as needed)
#     outer_iters_list = [100]        # maximum number of outer iterations
#     dataset_sizes = [1000,2000,5000, 10000]        # different dataset sizes
#     dimensions = [2, 5, 10, 50]                # different dimensions
    
#     experiment_results = {}
    
#     for outer_iters in outer_iters_list:
#         for ds in dataset_sizes:
#             for dim in dimensions:
#                 print(f"\nRunning experiment: outer_iters={outer_iters}, dataset_size={ds}, dimension={dim}")
#                 error_record = run_experiment(outer_iters, ds, dim, epsilon=epsilon)
#                 key = f"outer{outer_iters}_ds{ds}_dim{dim}"
#                 experiment_results[key] = error_record
                
#                 # Plot error evolution for this experiment
#                 plot_error_evolution(error_record, ds,outer_iters, dim)
#     return experiment_results

if __name__ == "__main__":
    results = run_all_experiments(epsilon=0.05)

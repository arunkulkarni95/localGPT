import time
import numpy as np

# Your existing import statements for loading the model and running the pipeline
from run_localGPT import load_model, retrieval_qa_pipline  # adjust according to your actual setup

# Define the sets of hyperparameters to test
context_window_sizes = [2048, 4096]
max_new_tokens_list = [512, 1024]
n_gpu_layers_list = [50, 100]
n_batch_list = [256, 512]
temperatures = [0.7, 1.0]
top_p_values = [0.9, 0.95]

# Initialize your model and pipeline here, if needed
device_type = "cuda"  # Change as per your hardware
model_id = "TheBloke/Llama-2-7b-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q4_K_M.gguf"
use_history = True
prompt_template_type = "llama"

# Benchmarking settings
num_runs = 5  # Number of runs for each hyperparameter setting
query = "What is the meaning of life?"

# Function to run the model and collect the benchmark score
def run_model(model, qa, num_runs=5):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        qa(query)  # Assuming that the qa function performs the entire operation you want to benchmark
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    return np.mean(times), np.std(times)

# Main loop for benchmarking
if __name__ == "__main__":
    for context_window_size in context_window_sizes:
        for max_new_tokens in max_new_tokens_list:
            for n_gpu_layers in n_gpu_layers_list:
                for n_batch in n_batch_list:
                    for temperature in temperatures:
                        for top_p in top_p_values:
                            print(f"Testing with context_window_size={context_window_size}, max_new_tokens={max_new_tokens}, n_gpu_layers={n_gpu_layers}, n_batch={n_batch}, temperature={temperature}, top_p={top_p}")

                            # Load the model and pipeline with the current set of hyperparameters
                            model = load_model(device_type, model_id, model_basename, n_gpu_layers, n_batch, context_window_size, max_new_tokens, temperature, top_p)
                            qa = retrieval_qa_pipline(device_type, use_history, prompt_template_type, n_gpu_layers, n_batch, context_window_size, max_new_tokens, temperature, top_p)

                            # Run the model and collect the benchmark score
                            avg_time, std_dev = run_model(model, qa, num_runs)
                            print(f"Average time: {avg_time:.4f}, Standard Deviation: {std_dev:.4f}")

import os
import subprocess

def run_mpi_program(file_path, num_nodes, processor_counts, mpi_script_path):
    """
    Run the MPI program with varying processor counts and record execution time.
    """
    results = {}
    for processors in processor_counts:
        command = [
            "mpiexec", "--oversubscribe", "-n", str(processors), "python3", mpi_script_path, file_path, str(num_nodes)
        ]
        try:
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            runtime = None
            for line in result.stdout.split("\n"):
                if "Execution Time:" in line:
                    runtime = float(line.split(":")[1].strip().split()[0])
                    break
            if runtime is not None:
                results[processors] = runtime
                print(f"[{processors} processors] Execution Time: {runtime:.2f} seconds")
            else:
                print(f"[{processors} processors] Failed to parse runtime.")
        except subprocess.CalledProcessError as e:
            print(f"[{processors} processors] Failed to get runtime.")
            print(f"Error:\n{e.stderr}")
    return results

def save_results(results, output_file):
    """
    Save the runtime results to a file.
    """
    with open(output_file, "w") as f:
        for processors, runtime in results.items():
            f.write(f"{processors} processors: {runtime:.2f} seconds\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Define paths and parameters
    mpi_script_path = "/Users/devvolve/Desktop/assignment 2/twitter_mpi.py"
    file_path = "/Users/devvolve/Desktop/assignment 2/twitter_combined.txt"  # Update with your dataset path
    num_nodes = 81306  # Number of nodes in the graph
    processor_counts = [2, 4, 8, 16]  # Adjust based on your system capabilities
    output_file = "twitter_mpi_runtime_results.txt"

    # Validate script and dataset paths
    if not os.path.exists(mpi_script_path):
        print(f"Error: MPI script not found at {mpi_script_path}")
        exit(1)
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        exit(1)

    # Run the performance study
    print("Starting MPI performance study...")
    results = run_mpi_program(file_path, num_nodes, processor_counts, mpi_script_path)
    print("Performance Study Results:", results)

    # Save results
    save_results(results, output_file)

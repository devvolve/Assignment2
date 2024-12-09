import subprocess
import re

def run_mpi_program(processor_count, file_path, num_nodes):
    """
    Run the MPI program with a given number of processors and capture the execution time.
    """
    command = [
        "mpiexec", "--oversubscribe", "-n", str(processor_count),
        "python3", "facebook_mpi.py", file_path, str(num_nodes)
    ]

    try:
        # Run the MPI command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Debug: Print raw output for troubleshooting
        print(f"[{processor_count} processors] Raw Output:\n{result.stdout}")

        # Extract runtime using regex
        runtime_match = re.search(r"Execution Time:\s*([\d.]+)\s*seconds", result.stdout)
        if runtime_match:
            return float(runtime_match.group(1))
        else:
            print(f"[{processor_count} processors] No runtime output found.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running MPI program with {processor_count} processors:")
        print(f"Command: {' '.join(command)}")
        print(f"Error Output:\n{e.stderr}")
        return None

def performance_study(file_path, num_nodes, processor_counts):
    """
    Perform performance study by running the MPI program with varying processor counts.
    """
    results = {}

    for count in processor_counts:
        print(f"Running with {count} processors...")
        runtime = run_mpi_program(count, file_path, num_nodes)

        if runtime is not None:
            results[count] = runtime
            print(f"[{count} processors] Runtime: {runtime:.2f} seconds")
        else:
            print(f"[{count} processors] Failed to get runtime.")

    return results

def main():
    # Define input parameters
    file_path = "path/to/facebook_combined.txt"  # Replace with actual path
    num_nodes = 4039  # Update with the number of nodes
    processor_counts = [2, 4, 8, 16]  # Processors to test

    print("Starting MPI performance study for Facebook dataset...")
    results = performance_study(file_path, num_nodes, processor_counts)

    # Save results to a file
    output_file = "facebook_mpi_runtime_results.txt"
    with open(output_file, "w") as f:
        for count, runtime in results.items():
            if runtime is not None:
                f.write(f"{count} processors: {runtime:.2f} seconds\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

def plot_results(results):
    processor_counts, runtimes = zip(*results)

    # Calculate speedup
    T1 = runtimes[0]
    speedup = [T1 / T for T in runtimes]

    # Calculate cost
    cost = [p * T for p, T in zip(processor_counts, runtimes)]

    # Plot runtime
    plt.figure()
    plt.plot(processor_counts, runtimes, marker='o')
    plt.xlabel("Number of Processors")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Processor Count")
    plt.grid()
    plt.show()

    # Plot speedup
    plt.figure()
    plt.plot(processor_counts, speedup, marker='o')
    plt.xlabel("Number of Processors")
    plt.ylabel("Speedup")
    plt.title("Speedup vs. Processor Count")
    plt.grid()
    plt.show()

    # Plot cost
    plt.figure()
    plt.plot(processor_counts, cost, marker='o')
    plt.xlabel("Number of Processors")
    plt.ylabel("Cost")
    plt.title("Cost vs. Processor Count")
    plt.grid()
    plt.show()


# Example usage after running performance_study
results = [(2, 30.5), (4, 16.2), (8, 8.1), (16, 4.1)]  # Replace with actual results
plot_results(results)

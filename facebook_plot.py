import matplotlib.pyplot as plt

# Sample data for Facebook dataset runtime results
# Replace these with actual results from the performance study
processor_counts = [2, 4, 8, 16]
execution_times = [120.0, 65.0, 35.0, 20.0]  # Replace with actual runtime values

# Calculate parallel speedup
speedup = [execution_times[0] / t for t in execution_times]

# Calculate parallel cost (Cost = Processors * Time)
cost = [p * t for p, t in zip(processor_counts, execution_times)]

# Plot runtime vs. processors
plt.figure(figsize=(10, 6))
plt.plot(processor_counts, execution_times, marker='o')
plt.title('Runtime vs. Processor Count (Facebook Dataset)')
plt.xlabel('Number of Processors')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
plt.xticks(processor_counts)
plt.savefig('facebook_runtime_vs_processors.png')
plt.show()

# Plot speedup vs. processors
plt.figure(figsize=(10, 6))
plt.plot(processor_counts, speedup, marker='o')
plt.title('Parallel Speedup vs. Processor Count (Facebook Dataset)')
plt.xlabel('Number of Processors')
plt.ylabel('Speedup')
plt.grid(True)
plt.xticks(processor_counts)
plt.savefig('facebook_speedup_vs_processors.png')
plt.show()

# Plot cost vs. processors
plt.figure(figsize=(10, 6))
plt.plot(processor_counts, cost, marker='o')
plt.title('Parallel Cost vs. Processor Count (Facebook Dataset)')
plt.xlabel('Number of Processors')
plt.ylabel('Cost (Processors × Time)')
plt.grid(True)
plt.xticks(processor_counts)
plt.savefig('facebook_cost_vs_processors.png')
plt.show()

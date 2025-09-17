import numpy as np
import matplotlib.pyplot as plt

# Parameters
time_steps = 100
threshold = 1.0
leak_factor = 0.9
input_current = np.random.uniform(0.1, 0.2, time_steps)
membrane_potential = np.zeros(time_steps)
prefire_membrane_potential = np.zeros(time_steps)
spikes = np.zeros(time_steps)

# Simulation
for t in range(1, time_steps):
    membrane_potential[t] = leak_factor * membrane_potential[t-1] + input_current[t]
    prefire_membrane_potential[t] = membrane_potential[t]
    if membrane_potential[t] >= threshold:
        spikes[t] = 1
        membrane_potential[t] = 0  # Reset after spike

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(membrane_potential, label="Membrane Potential")
plt.plot(prefire_membrane_potential, label="Membrane Potential (Before Firing)")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.legend()
plt.ylabel("Voltage")
plt.title("LIF Neuron Simulation")
plt.subplot(2, 1, 2)
plt.stem(spikes, label="Spikes", use_line_collection=True)
plt.xlabel("Timesteps")
plt.ylabel("Spike out")
plt.legend()
plt.show()
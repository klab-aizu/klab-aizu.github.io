import numpy as np
import matplotlib.pyplot as plt

# Parameters
time_steps = 1000
dt = 0.5
a, b, c, d = 0.02, 0.2, -65, 8
v = -65
u = b * v
threshold = 30
input_current = np.zeros(time_steps)
input_current[100:800] = 10  # Step input

# Storage for plotting
v_trace = []
u_trace = []
spikes = []

# Simulation
for t in range(time_steps):
    v_trace.append(v)
    u_trace.append(u)
    if v >= threshold:
        v = c
        u += d
        spikes.append(1)
    else:
        spikes.append(0)
        dv = 0.04 * v**2 + 5 * v + 140 - u + input_current[t]
        du = a * (b * v - u)
        v += dv * dt
        u += du * dt

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(v_trace, label="Membrane Potential (v)")
plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
plt.ylabel("Voltage")
plt.legend()
plt.subplot(2, 1, 2)
plt.stem(spikes, linefmt="r-", markerfmt="ro", basefmt="k-", label="Spikes")
plt.xlabel("Timesteps")
plt.ylabel("Spike out")
plt.legend()
plt.show()

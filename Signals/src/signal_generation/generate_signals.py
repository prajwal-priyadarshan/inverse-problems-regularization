import numpy as np
import matplotlib.pyplot as plt
def sinusoid(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)


def multisine(t: np.ndarray, freqs=(2, 5, 9), amps=(1.0, 0.6, 0.3)) -> np.ndarray:
    out = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        out += a * np.sin(2 * np.pi * f * t)
    return out


def piecewise(t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    thirds = len(t) // 3
    out[:thirds] = 1.0
    out[thirds:2 * thirds] = -0.5
    out[2 * thirds:] = 0.7
    return out

# Generate signals
t = np.linspace(0, 1, 1000)
y1 = sinusoid(t)
y2 = multisine(t)
y3 = piecewise(t)

# Plot all signals
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(t, y1)
plt.title("Sinusoid")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(t, y2)
plt.title("Multisine")
plt.xlabel("Time")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(t, y3)
plt.title("Piecewise")
plt.xlabel("Time")
plt.grid(True)

plt.tight_layout()

# Save for PPT
plt.savefig("three_signals_subplot_1x3.png", dpi=300, bbox_inches="tight")
plt.show()
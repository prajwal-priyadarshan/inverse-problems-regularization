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


def sine_mixture_2d(
    size: int = 64,
    components=None,
    normalize: bool = True,
) -> np.ndarray:
    """Create a 2D signal as a mixture of spatial sine waves.

    Args:
        size: Output image size (size x size).
        components: List of tuples (amplitude, fx, fy, phase).
        normalize: If True, scale output to [0, 1].

    Returns:
        2D numpy array representing the generated signal.
    """
    if components is None:
        components = [
            (1.00, 2.0, 1.0, 0.0),
            (0.65, 5.0, 3.0, np.pi / 4),
            (0.40, 9.0, 4.0, np.pi / 2),
        ]

    x = np.linspace(0.0, 1.0, size, endpoint=False)
    y = np.linspace(0.0, 1.0, size, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    signal = np.zeros((size, size), dtype=float)
    for amp, fx, fy, phase in components:
        signal += amp * np.sin(2.0 * np.pi * (fx * xx + fy * yy) + phase)

    if normalize:
        s_min, s_max = signal.min(), signal.max()
        if s_max > s_min:
            signal = (signal - s_min) / (s_max - s_min)

    return signal


if __name__ == "__main__":
    # Quick visual check when running this module directly.
    t = np.linspace(0, 1, 1000)
    y1 = sinusoid(t)
    y2 = multisine(t)
    y3 = piecewise(t)

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
    plt.savefig("three_signals_subplot_1x3.png", dpi=300, bbox_inches="tight")
    plt.show()
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    n_samples: int = 200
    gaussian_sigma: float = 2.5
    kernel_radius: int = 10
    noise_levels: dict = None

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = {
                "low": 0.001,
                "medium": 0.01,
                "high": 0.05,
            }

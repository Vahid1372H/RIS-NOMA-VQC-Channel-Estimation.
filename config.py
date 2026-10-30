"""Simulation and optimization settings aligned with Table III and Algorithm 1."""
from dataclasses import dataclass, field, replace
from typing import Tuple
import numpy as np

@dataclass
class SimulationConfig:
    # Table III: core deployment
    M: int = 2
    Nt: int = 1
    L: int = 64
    d_sr_m: float = 30.0
    user_distances_m: Tuple[float, ...] = (40.0, 60.0)
    d0_m: float = 1.0
    pathloss_bs_ris: float = 2.2
    pathloss_ris_user: float = 2.8
    displacement_m_per_block: float = 0.002
    rician_K: float = 3.0
    f_D: float = 0.05
    rho: float = 0.975  # Table III rounded value; alternatively derive via J0.
    N_p: int = 4
    T: int = 6000
    W: int = 10
    snr_db: Tuple[float, ...] = tuple(float(x) for x in range(-5, 31, 5))
    # NOMA / modulation parameters reported for the BER experiment
    power_shares: Tuple[float, ...] = (0.2, 0.8)
    modulation: str = "QPSK"
    symbols_per_block: int = 64
    # VQC baseline
    n_qubits: int = 4
    circuit_depth: int = 3
    # Algorithm 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_gap: int = 10
    learning_rate: float = 3e-3
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 8
    seeds: Tuple[int, ...] = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101, 113, 127, 139, 151, 163)
    device: str = "cpu"
    torch_num_threads: int = 3

    def __post_init__(self):
        if self.M < 1 or self.L < 1 or self.T <= self.W + 20:
            raise ValueError("M,L must be positive and T must comfortably exceed W.")
        if len(self.user_distances_m) < self.M:
            raise ValueError("Provide one RIS-user distance per user.")
        if not np.isclose(sum(self.power_shares[:self.M]), 1.0):
            raise ValueError("NOMA power shares for the active users must sum to one.")
        if not np.isclose(self.train_fraction+self.validation_fraction+self.test_fraction, 1.0):
            raise ValueError("Chronological split fractions must sum to one.")
        if self.N_p < 1 or self.W < 1 or self.n_qubits < 1:
            raise ValueError("N_p, W and n_qubits must be positive.")

    @property
    def jakes_rho(self) -> float:
        """Jakes coefficient J0(2 pi f_D); Table III lists the rounded 0.975."""
        from scipy.special import j0
        return float(j0(2.0*np.pi*self.f_D))

    def with_overrides(self, **kwargs):
        return replace(self, **kwargs)

DEFAULT_CONFIG = SimulationConfig()

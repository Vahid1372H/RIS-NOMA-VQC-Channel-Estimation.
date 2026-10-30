# RIS–NOMA Channel Estimation Simulation with VQC

This package is a self-contained research implementation of the core simulation described in “Assessing the Impact of Variational Quantum Circuits on Cascaded Channel Estimation in 6G RIS-NOMA Systems.” The default parameters are aligned with Table III, while the training protocol, temporal data splits, and early stopping follow Algorithm 1. Code explanations and comments are written in English and Persian for readability.

## Features

- Generation of cascaded BS–RIS–user channels with Rician fading, a line-of-sight (LOS) component, fixed RIS phases, and Gauss–Markov evolution using a Jakes coefficient.
- Generation of pilot observations and QPSK/NOMA symbols; LS, LMMSE, and Kalman estimators.
- CNN-LSTM; a four-qubit VQC head with RY encoding, ring CNOT entanglement, and Pauli-Z expectation readout; and a classical ablation with a matched parameter budget.
- NMSE and R² estimation, paired Wilcoxon / t-tests, and a TOST framework applied to dB differences at the seed level.
- A small Monte Carlo run for pipeline validation, with checkpoint and metrics output.

## Installation

Python 3.10 or later is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

From the package's parent directory, run:

```bash
python -m ris_noma_vqc_simulation.main --output results
```

By default, the quick run uses two seeds, two SNR points, and T=600 blocks for execution validation. Outputs are saved to `results/metrics.json` and `.pt` checkpoint files.

## Running the Full Table III Configuration

```bash
python -m ris_noma_vqc_simulation.main --full --output results_full
```

The default configuration includes M=2, L=64, f_D=0.05, rho=0.975, N_p=4, T=6000, W=10, an SNR grid from −5 to 30 dB in 5 dB increments, and 15 seeds. A full run and training multiple models may take considerable time. For custom experiments, instantiate or override `SimulationConfig` in `config.py`, or call `run()` from `main.py`.

## Reproducibility and Modeling Notes

1. The seed deterministically initializes the channel and pilot noise. For fair comparisons, models are evaluated on the same trace.
2. Data are split chronologically into train/validation/test sections, with a gap of size W between sections. Windows do not cross section boundaries, and standardization statistics are computed only from training windows.
3. The synthetic channel is constructed from the assumptions in the paper. No field-measurement dataset is provided in the paper or the input files. Accordingly, this package generates reproducible synthetic data, not laboratory measurements.
4. The VQC simulator is an exact, differentiable PyTorch state-vector simulator. The `parameter_shift_jacobian` method is provided to verify the parameter-shift rule; by default, training uses exact automatic differentiation through the same circuit. For an independent PennyLane check (PennyLane is not a required dependency), a small circuit can be adapted in a separate research environment.
5. Table III rounds rho=0.975 and also introduces the formula J0(2πf_D). The default configuration uses the explicit table value; `cfg.jakes_rho` provides the value computed from f_D.
6. Additional paper details (all sweeps, BER/OFDMA-level reporting, QLSTM gate-level ablation, and hardware noise) are outside the scope of this core implementation. This code does not claim to numerically reproduce all results in the paper, and implements path-loss parameters using explicit, editable assumptions.

## Project Structure

```text
ris_noma_vqc_simulation/
  config.py          # Table III parameters
  channel_model.py   # Channel, pilots, NOMA, and analytical estimators
  qnn_models.py      # CNN-LSTM, VQC, and ablation
  train_evaluate.py  # Temporal split and Algorithm 1
  stats_analysis.py  # Metrics and paired tests
  main.py            # Monte Carlo orchestrator
```

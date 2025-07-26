# Pilot Allocation Simulator

This repository contains a single Python script (`pa.py`) that implements a set of algorithms for allocating pilot sequences in wireless networks. The code was written for simulations with a cell-free massive MIMO system, evaluating pilot assignment heuristics and the resulting user rates.

## Requirements

The script relies on a few scientific Python packages. Install them with `pip`:

```bash
pip install numpy scipy cvxpy matplotlib
```

The power control optimisation uses the GUROBI solver through CVXPY. A licensed installation of GUROBI is required to reproduce the numerical results.

## Usage

Run the simulator from the command line passing the following four parameters:

1. **M** – number of access points
2. **K** – number of user equipments
3. **P** – number of available pilot sequences
4. **RODADAS** – number of independent simulation rounds

Example:

```bash
python pa.py 16 40 10 100
```

Outputs are written to text files whose names embed the parameter values. Each algorithm creates a CDF file (`cdf_M_K_P_<algo>.dat`) and the overall results are appended to `res_M_K_P.dat`.

## Algorithms

The script currently runs two pilot allocation strategies:

- **maxkcut** – heuristic based on a maximum-k-cut approach;
- **basic** – a simple baseline allocation.

Other algorithms (random, greedy, ibasic, wgf and wgfsb) are implemented in the code but commented out in the configuration dictionary at the end of the script. They can be enabled by editing the `algo` dictionary.

## Repository Contents

- `pa.py` – main simulation script containing the implementation of all algorithms.
- `README.md` – this documentation file.

Feel free to adapt the simulator for your experiments or extend it with additional allocation strategies.

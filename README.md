# ICU Foundation Model Demo (GPT-MEDIC Simulation)

This repository serves as a proof-of-concept simulation for pre-training foundation models on multi-center ICU event streams. It demonstrates time-series sequence preprocessing, transformer-based self-supervised training, and a decentralized federated averaging loop.

## Project Structure
- `data/`: Generation scripts for synthetic multi-center ICU continuous event streams.
- `models/`: PyTorch implementation of a Time-Series Transformer block.
- `federated/`: Simulations of decentralized multi-hospital weight aggregation (FedAvg).
- `train.py`: Main self-supervised pre-training loop.

## Conceptual Framework
Inspired by architectures like [Vector Institute's Odyssey](https://github.com/VectorInstitute/odyssey) and [Rätsch Lab's ICareFM](https://github.com/ratschlab/icarefm), this demo handles the irregular temporal spacing and high variance characteristic of critical care data.

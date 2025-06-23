# Forensic Analysis of Network Attacks using Graph-Based Modeling

![Network Security](https://img.shields.io/badge/domain-network%20security-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red)

A machine learning system for detecting network attacks using Graph Neural Networks (GNNs) with FastAPI backend.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Features ✨

- **Graph-Based Analysis** - Models network traffic as graphs (nodes=hosts, edges=connections)
- **GAT Model** - Graph Attention Network for attack classification
- **Community Detection** - Identifies attack patterns using:
  - Louvain method
  - Spectral clustering
- **Web Interface** - Interactive visualization of results
- **PCAP Processing** - Supports real network capture files

## Installation 🛠️

```bash
# Clone repository
git clone https://github.com/yourusername/Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling.git
cd Forensic-Analysis-of-Network-Attacks-using-Graph-Based-Modeling

# Create conda environment
conda create --name netsec python=3.9
conda activate netsec

# Install dependencies
pip install -r requirements.txt

# Install PyG components
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# Install tshark (Ubuntu)
sudo apt-get install tshark

```

## Training the Model 🧠

The best model comes already in the ```/backend``` folder, but if you want to train a own model based on the same partitions but changing hyperparameters, here are the instructions:

Prepare data partitions in ```data_partitions``` (download in https://drive.google.com/drive/folders/1l-tnTSyOzWmW3Qu_1qQWbBfPXwFMEVB_?usp=drive_link) with:

1. Network traffic npz files which is a balanced version of the BCCC2018 dataset.

2. Required features (see ```config.py``` )

Run training:

```bash
python main.py
```
Keep config parameters (```config.py```):

```python
DATA_FOLDER = "CSVs/BCCC-CIC2018"  # Dataset path
MAX_GRAPH_NODES = 5000             # Max nodes per graph
EPOCHS = 100                       # Training iterations
LEARNING_RATE = 0.001              # Model learning rate
```

## Running the API 🚀
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 3600
```
Access frontend at: ```http://localhost:8000```

API Endpoints

| **Endpoint**  |	**Description** |
| ------------- | ------------------ |
| ```/api/upload```  |	Upload PCAP files |
| ```/api/analyze-communities```  |	Community detection |
| ```/api/merge```  |	Combine file chunks |


## Project Structure 📂

```text
backend/
├── api.py               # FastAPI endpoints
├── classifier.py        # Attack classifier
├── graph_algorithms.py  # Community detection
├── best_model.pt        # Best trained model achieved
└── pcap_processor.py    # PCAP processing
frontend/
├── scripts.js           # JS frontend
├── index.html           # Index
├── styles.css           # Styles 
CSVs/                    # Dataset directory
models/                  # GNN models
utils/                   # Support modules
config.py                # Configuration
main.py                  #Main
train.py                 # Training pipeline
```

### Troubleshooting 🚑

| **Issue**  |	**Solution** |
| ------------- | ------------------ |
| CUDA OOM  |	Reduce ```MAX_GRAPH_NODES``` or ```BATCH_SIZE``` |
| PCAP errors  |	Verify ```tshark``` installation |
| Missing deps  |	```pip install --upgrade -r requirements.txt``` |
| API timeouts  |	Increase ```--timeout-keep-alive``` value|




For additional support, please open an issue in the repository.
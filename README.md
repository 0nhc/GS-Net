# GS-Net
This project aims to address issues encountered during the migration of the repository [GS-Net](https://github.com/graspnet/graspness_unofficial) to an RTX 4090 GPU.
The original repo is a fork of paper "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021) by [Zibo Chen](https://github.com/rhett-chen).

## Prerequisites
* CUDA 12.1
* Anaconda

## Installation
* Create Virtual Environment
    ```sh
    conda create -n gsnet python=3.8
    ```

* Install GS-Net
    ```sh
    git clone https://github.com/0nhc/GS-Net.git
    cd GS-Net
    conda activate gsnet
    pip install -r requirements.txt
    cd pointnet2
    python setup.py install
    cd ..
    cd knn
    python setup.py install
    cd ..
    cd graspnetAPI
    pip install .
    ```

* Install MinkowskiEngine
    ```sh
    git clone --recursive https://github.com/0nhc/MinkowskiEngine.git
    cd MinkowskiEngine
    conda activate gsnet
    pip install -r requirements.txt
    python setup.py install
    ```

* Download Checkpoint

    Download from [https://drive.google.com/file/d/1ZYCKAf6EF7aghWDbYAqelZWK0cVn8LGX/view?usp=sharing](https://drive.google.com/file/d/1ZYCKAf6EF7aghWDbYAqelZWK0cVn8LGX/view?usp=sharing).
    Then replace `checkpoint_path` in [config/gsnet_flask_server.yaml](config/gsnet_flask_server.yaml)

## Quick Start
* Start Server
```sh
# Terminal 1
conda activate gsnet
python flask_server.py
```
* Try Demo Client
```sh
# Terminal 2
conda activate gsnet
python flask_client.py
```

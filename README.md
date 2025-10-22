# GS-Net
This project aims to address issues encountered during the migration of the repository [GS-Net](https://github.com/graspnet/graspness_unofficial) to RTX 40 Series GPUs.

The original repo is a fork of paper "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021) by [Zibo Chen](https://github.com/rhett-chen).

## Prerequisites
My system configurations:
* RTX 40 Series GPU
* Ubuntu 22.04
* NVIDIA 535 Driver
* CUDA 12.1
* Python 3.10

## Installation
* Create Virtual Environment
    ```sh
    sudo apt install python-is-python3 python3.10-venv python3.10-dev libopenblas-dev
    python3 -m venv ~/venv/gsnet
    ```

* Install GS-Net
    ```sh
    git clone https://github.com/0nhc/GS-Net.git
    cd GS-Net
    source  ~/venv/gsnet/bin/activate
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
    source  ~/venv/gsnet/bin/activate
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
source  ~/venv/gsnet/bin/activate
python flask_server.py
```
* Try Demo Client
```sh
# Terminal 2
source  ~/venv/gsnet/bin/activate
python flask_client.py
```

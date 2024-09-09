# PINN-GNN Traffic Flow Prediction
![Loading Model Overview](https://github.com/viviRG2024/PINNGNN/blob/main/asset/framework.jpg "Model Overview")
---

This repository contains the implementation of PINN-GNN (Physics-Informed Graph Neural Networks), an innovative approach to traffic forecasting under flooding that combines the power of Graph Neural Networks with the principles of Physics-Informed Neural Networks. 

Our proposed model has been rigorously tested on various traffic datasets, demonstrating superior performance in: 1) **Accuracy** of traffic flow predictions 2) **Robustness** to missing or noisy data 3) **Generalization** to unseen traffic scenarios

**About the Model**

PINN-GNN addresses the complex task of traffic forecasting by integrating physical laws and constraints into the learning process of graph neural networks. Our model uniquely features:

1. **Graph Neural Networks (GNN)**: To capture the spatial dependencies in the traffic network.

2. **Physics-Informed Neural Networks (PINN)**: To incorporate traffic flow dynamics and physical constraints.

3. **Field Effect Module**: To model long-range dependencies and global traffic patterns.

Full text (PDF) is available at [updating]().

## Dependencies

### OS

Linux systems (*e.g.* Ubuntu and CentOS). 

### Python

The code is built based on Python 3.9, PyTorch 1.13.0, and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). 

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended to create a virtual python environment.

We implement our code based on [BasicTS](https://github.com/zezhishao/BasicTS/tree/master).

### Other Dependencies

```bash
pip install -r requirements.txt
```

## Getting started

### Download Data

You can download data from [google drive](https://drive.google.com/drive/folders/1F7vzXht4A9tWgiX5Jens-vGLirgVVGti?usp=drive_link) and unzip it.

### Preparing Data


- **Pre-process Data**
The further description about data pre-processing can be found in [data_preparation](/procedure.md).
After that, you can pre-process all datasets by

    cd /path/to/your/project
    bash scripts/data_preparation/all.sh

Then the `dataset` directory will look like this:

```text
datasets
   ├─PEMS04
   ├─PEMS08
   ├─ cities
   |    ├─ augsburg
   |    ├─ cagliari
   |    ├─ darmstadt
   |    ├─ essen
   |    ├─ hamburg
   |    ├─ innsbruck
   |    ├─ london
   |    ├─ luzern
   |    ├─ madrid
   |    ├─ manchester
   |    ├─ marseille
   |    ├─ paris
   |    ├─ strasbourg
   |    ├─ taipeh
   |    └─ toronto
   ├─README.md
```

### Pre-Training on Models

```
cd /path/yourproject
```

Then run the folloing command to run in Linux screen.

```
screen -d -m python pinngnn/run.py --cfg='pinngnn/PEMS04.py' --gpus='0'

screen -d -m python pinngnn/run.py --cfg='pinngnn/PEMS08.py' --gpus='0'

```

### Downstream Predictor

After pre-training , copy your pre-trained best checkpoint to `mask_save/`.
For example:


```bash
cp checkpoints/pinngnn/064b0e96c042028c0ec44856f9511e4c/pinngnn_best_val_MAE.pt mask_save/pinngnn_PEMS04_864.pt
cp checkpoints/pinngnn/50cd1e77146b15f9071b638c04568779/pinngnn_best_val_MAE.pt mask_save/pinngnn_PEMS08_864.pt
```

Then run the predictor as :

```
screen -d -m python pinngnn/run.py --cfg='pinngnn/PEMS04.py' --gpus='0' 

screen -d -m python pinngnn/run.py --cfg='pinngnn/PEMS08.py' --gpus='0'
```

* To find the best result in logs, you can search `best_` in the log files.

## Citation
If you use this code in your project, please consider citing the following paper:
```bibtex
@article{ updating
}
```

## License
Please see the [license](LICENSE) for further details.

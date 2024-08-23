import os
import sys
import torch
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae,pinngnn_physics_loss
from basicts.utils import load_adj

from .arch import PINNsGNN

CFG = EasyDict()

CFG.DESCRIPTION = "PINNs-GNN model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "PINNsGNN"
CFG.MODEL.ARCH = PINNsGNN
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "normlap")
adj_mx = torch.Tensor(adj_mx[0])
CFG.MODEL.PARAM = {
    "Ks": 3,
    "Kt": 3,
    "blocks": [[3, 32, 64], [64, 32, 128], [128, 64, 256], [256, 128], [1]],
    "T": 12,
    "n_vertex": 307,
    "act_func": "glu",
    "graph_conv_type": "cheb_graph_conv",
    "gso": adj_mx,
    "bias": True,
    "droprate": 0.5
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]  # 假设我们使用流量、占有率和速度
CFG.MODEL.TARGET_FEATURES = [0]

CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = pinngnn_physics_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 100],
    "gamma": 0.5
}

CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as DATA

import torchvision.models.video as vmodels
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

from pytorch_lightning.core.optimizer import LightningOptimizer
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint
import torchmetrics as metrics

from pytorchvideo.data import UniformClipSampler as UCS
from pytorchvideo.data import LabeledVideoDataset as LVDS
from pytorchvideo import models
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union
import warnings
from sklearn import preprocessing
import numpy as np
import pickle as pkl
import csv
import os
import os.path as path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision

import json # for reading from json file
import glob # for listing files inside a folder
from torchmetrics import JaccardIndex

gt_loaded_tensor = torch.load('./gt_tensor_stacked.pth')
pred_loaded_tensor = torch.load('./pred_tensor_stacked.pth')

jaccard = JaccardIndex(task="multiclass", num_classes=49)
print(jaccard(gt_loaded_tensor, pred_loaded_tensor))
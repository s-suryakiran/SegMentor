import gzip
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'moving_mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

def load_fixed_set(root):
    # Load the fixed dataset
    # filename = 'moving_mnist/mnist_test_seq.npy'
    # path = os.path.join(root, filename)
    dataset = np.load(root)
    return dataset

def load_dynamicset(test_imgs):
    file_links = os.listdir(test_imgs)
    file_images = [file for file in file_links if file.endswith(('.jpg', '.jpeg', '.png'))]
    npy_array = np.zeros((22,1,160,240,3))
    for img_idx,f in enumerate((file_images)):
        img = Image.open(test_imgs+'/'+f)
        img_resized = img.resize((240, 160))
        img_resized
        npy_array[img_idx, 0] = np.array(img_resized)

    return npy_array


class ShapesDataset(Dataset):
    """Shapes Dataset"""

    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11,transform=None):
        super(ShapesDataset, self).__init__()

        self.dataset = None
        if(is_train):
            self.dataset = load_fixed_set(root)
        else:
            self.dataset = load_dynamicset(root)
        self.length = self.dataset.shape[1]

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.mean = None
        self.std = None

    def __getitem__(self, idx):
        
        images = self.dataset[:, idx, ...]

        images = np.moveaxis(images, -1, 1)

        frames_input = images[:self.n_frames_input]
        
        output = images[self.n_frames_input:]

        frames_input = torch.from_numpy(frames_input / 255.0).contiguous().float()
        output = torch.from_numpy(output / 255.0).contiguous().float()
        
        return frames_input, output

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root,
              num_workers=4, pre_seq_length=11, aft_seq_length=11,mode = "train"):
    if(mode=="train"):
        train_set = ShapesDataset(root="./train_dataset.npy")
        test_set = ShapesDataset(root="./test_dataset.npy")
        print(val_batch_size,batch_size)
        dataloader_train = torch.utils.data.DataLoader(train_set,
                                                       batch_size=batch_size, shuffle=True,
                                                       pin_memory=True, drop_last=True,
                                                       num_workers=num_workers)
        dataloader_vali = torch.utils.data.DataLoader(test_set,
                                                      batch_size=batch_size, shuffle=False,
                                                      pin_memory=True, drop_last=True,
                                                      num_workers=num_workers)
        dataloader_test = torch.utils.data.DataLoader(test_set,
                                                      batch_size=batch_size, shuffle=False,
                                                      pin_memory=True, drop_last=True,
                                                      num_workers=num_workers)
    else:
        test_set = ShapesDataset(root=data_root,is_train=False)
        dataloader_train = torch.utils.data.DataLoader(test_set,
                                                       batch_size=1, shuffle=False,
                                                       pin_memory=True, drop_last=True,
                                                       num_workers=num_workers)
        dataloader_vali = torch.utils.data.DataLoader(test_set,
                                                      batch_size=1, shuffle=False,
                                                      pin_memory=True, drop_last=True,
                                                      num_workers=num_workers)
        dataloader_test = torch.utils.data.DataLoader(test_set,
                                                      batch_size=1, shuffle=False,
                                                      pin_memory=True, drop_last=True,
                                                      num_workers=num_workers)

    return dataloader_train, dataloader_vali, dataloader_test
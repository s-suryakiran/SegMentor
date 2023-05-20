import numpy as np
import torch
import torchvision

import json # for reading from json file
import glob # for listing files inside a folder
from PIL import Image, ImageDraw # for reading images and drawing masks on them.


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import getopt, sys

import argparse


# Create a custom dataset class for Primitive Shapes dataset subclassing PyTorch's Dataset class
class PrimitiveShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root,transforms):
        self.transforms = transforms

        imgs = glob.glob(root + '/*/*.png')
        self.imgs = sorted(imgs)
        
        # Mask data is stored in a json file
        masks = glob.glob(root + '/*/*.npy')
        self.masks = sorted(masks)  
        
        replicated_list = [item for item in self.masks for _ in range(22)]
        
        self.masks = replicated_list


        
    def __getitem__(self, idx):
        # Have already aligned images and JSON files; can now
        # simply use the index to access both images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        
        image_id = img_path.split("/")[-1].split("_")[-1][:-4]
        
#         print(image_id)
        
        mask_id = int(image_id)
        
        # Read image using PIL.Image and convert it to an RGB image
        img = Image.open(img_path).convert("RGB")
        
        mask_img = np.load(mask_path)[mask_id]
        
        all_classes = np.unique(mask_img)
        
        
        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
            
        array_length = 49
        one_d_array = np.zeros(array_length,dtype=np.float32)
        
        one_d_array[all_classes] = 1

        return img, mask_img,one_d_array[1:]

    
    def __len__(self):
        return len(self.imgs)


def test_model( traindataset_folder , val_dataset_folder, model_weights_path):


    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PrimitiveShapesDataset(traindataset_folder, train_transform)
    val_dataset = PrimitiveShapesDataset(val_dataset_folder, val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load a empty ResNet-50 model
    model = resnet50(pretrained=False)

    # Modify the last layer to output 48 units with sigmoid activation as there are multiple classes in a single image
    num_classes = 48  # Only 48 classes for resnet training because not background for classification
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes)#,
    #     nn.Sigmoid()
    )

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # model = model.to(device)

    def calculate_accuracy(outputs, labels):
        preds = outputs > 0.5
        return (preds == labels).float().mean()

    def get_accuracy(model, dataloader):
        model.eval()
        correct = 0
        total = 0
    #     accuracy = 0

        with torch.no_grad():
            for data in dataloader:
                inputs,mask_image, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predictions = torch.round(outputs)
                
                total += labels.size(0)
    #             print(labels.size(0))
                correct += calculate_accuracy(outputs, labels).item()
    #             print("hello",correct)
    #             print("hello",correct)
                
        accuracy = (100 * correct)/len(dataloader) 
        return accuracy

    # Load the model weights
    loaded_weights = torch.load(model_weights_path)
    model.load_state_dict(loaded_weights)
    model.to(device)

    # Get the training and validation accuracy

    print("calculating train accuracy ....")
    train_accuracy = get_accuracy(model, train_dataloader)
    print("calculating test accuracy ....")
    val_accuracy = get_accuracy(model, val_dataloader)

    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")

 
# Create an argument parser object
parser = argparse.ArgumentParser()

# Add the arguments for the train and test folders
parser.add_argument('--train', help='Path to the training dataset')
parser.add_argument('--val', help='Path to the testing dataset')
parser.add_argument('--weights_path', help='Path to the testing dataset')

# Parse the command-line arguments
args = parser.parse_args()


if (not args.train) or (not args.val) or (not args.weights_path) :
    print("Expected command line arguments not found so exiting the program")

# Store the training and testing dataset paths in variables
train_folder = args.train
test_folder = args.val
weights_path = args.weights_path

# print(train_folder)
# print(test_folder)


test_model(train_folder, test_folder, weights_path)

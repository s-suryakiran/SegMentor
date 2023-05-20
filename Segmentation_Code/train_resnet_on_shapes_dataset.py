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


def train(traindataset_folder, val_dataset_folder):


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
    model = model.to(device)

    def calculate_accuracy(outputs, labels):
        preds = outputs > 0.5
        return (preds == labels).float().mean()

    prev_best_val_accuracy = 0.0

    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, (inputs,mask_img, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels).item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = running_acc / len(train_dataloader)

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0

        with torch.no_grad():
            for i, (inputs,mask_img, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                running_val_acc += calculate_accuracy(outputs, labels).item()

        val_loss = running_val_loss / len(val_dataloader)
        val_acc = running_val_acc / len(val_dataloader)
        
        if val_acc > prev_best_val_accuracy:
            prev_best_val_accuracy = val_acc
            model_weights_path = "resnet50_best_model.pth"
            torch.save(model.state_dict(), model_weights_path)
            print("model reached best val accuracy in this epoch, so saving the the model to resnet50_best_model.pth")

        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")



 
# Create an argument parser object
parser = argparse.ArgumentParser()

# Add the arguments for the train and test folders
parser.add_argument('--train', help='Path to the training dataset')
parser.add_argument('--val', help='Path to the testing dataset')

# Parse the command-line arguments
args = parser.parse_args()


if (not args.train) or (not args.val):
    print("Expected command line arguments not found so exiting the program")

# Store the training and testing dataset paths in variables
train_folder = args.train
test_folder = args.val

# print(train_folder)
# print(test_folder)


train(train_folder, test_folder)




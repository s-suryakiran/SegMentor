import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch.segmentation_models_pytorch as smp
import torch.nn as nn

from pprint import pprint
from torch.utils.data import DataLoader

import numpy as np
import torch
import torchvision

import json # for reading from json file
import glob # for listing files inside a folder
from PIL import Image, ImageDraw # for reading images and drawing masks on them.
import torchvision.transforms as transforms
import os
import torchmetrics

import argparse


# Create a custom dataset class for Nature
# dataset subclassing PyTorch's Dataset class
class PrimitiveShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root1,root2,transforms,mask_transforms):
        self.transforms = transforms
        self.mask_transforms = mask_transforms

        self.imgs = glob.glob(root1 + '/*.png')
        
        self.root1 = root1
        self.root2 = root2
        


        
    def __getitem__(self, idx):
        # Have already aligned images and JSON files; can now
        # simply use the index to access both images and masks
        img_path = self.imgs[idx]
        
        mask_path = self.root2 +"/" + self.imgs[idx].split("/")[-1]
        
        # Set the desired padding width
        padding_width = 16

        # Create a new image with the desired width and the same height as the original image
        new_image = Image.new('RGB', (240 + padding_width, 160))

        # Paste the original image onto the new image
        new_image.paste(Image.open(img_path), (0, 0))
            
        
        # Read image using PIL.Image and convert it to an RGB image
        img = np.moveaxis(np.array(new_image) ,  -1, 0)
        
        
        # Set the desired padding width
        mask_padding_width = 16

        # Create a new image with the desired width and the same height as the original image
        mask_new_image = Image.new('L', (240+mask_padding_width, 160))
        
#         print(np.unique(Image.open(mask_path)))

        # Paste the original image onto the new image
        mask_new_image.paste(Image.open(mask_path), (0, 0))
        
        mask_img = np.array(mask_new_image)
        
#         print("tester",mask_img.max())
        
        all_classes = np.unique(mask_img)
        
        sample = dict(image=img, mask=mask_img)
        # Apply transforms
#         if self.transforms is not None:
#             img = self.transforms(img)
#             mask_img = torch.tensor(np.array(self.mask_transforms(mask_img)))
            

        return sample
    
    def __len__(self):
        return len(self.imgs)


class PetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name,encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights= encoder_weights, in_channels=in_channels, classes=out_classes, **kwargs, 
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained="customnet")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        
        self.softmaxfunc = nn.Softmax(dim=1)
        
        self.metrics = torchmetrics.IoU(num_classes=out_classes)
        
        self.no_of_classes = out_classes

    def forward(self, image):
        image = image/255.0
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        
        
        h, w = image.shape[2:]
#         print("hello",h,w)
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"].to(torch.int64)
        
#         print(torch.unique(mask), mask.shape)
        
#         print(mask.shape, "lll")
        
        
#         for i in range(1,49):
#             mask[mask == i] = i
            
#         print

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
#         assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
#         assert mask.max() <= 1.0 and mask.min() >= 0

#         print("hello", mask.max())
        
        

        logits_mask = self.forward(image)
        
#         print("hello hello", logits_mask.shape, mask.shape)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        
        

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = self.softmaxfunc(logits_mask)
        pred_mask = torch.argmax(prob_mask, 1, keepdim=True)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask[:,0,...].long(), mask.long(), 
                                               mode="multiclass", num_classes=self.no_of_classes)


        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)
    

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0006)


def testdeeplabv3plus( train_images, train_annotations, val_images, val_annotations, model_path):

    # init train, val, test sets
    train_dataset = PrimitiveShapesDataset(train_images, train_annotations,None,None)
    valid_dataset = PrimitiveShapesDataset(val_images,val_annotations,None,None)


    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)


    model = PetModel("DeepLabV3plus", "resnet50","customnet", in_channels=3, out_classes=49)

    trainer = pl.Trainer(
        gpus = 1,
        max_epochs=20,
    #     precision=16
    )

    # trainer.fit(
    #     model, 
    #     train_dataloaders=train_dataloader, 
    #     val_dataloaders=valid_dataloader,
    # )


    # good_deeplabv3plus_final_scaled.pth

    model.load_state_dict(torch.load(model_path))

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

# Create an argument parser object
parser = argparse.ArgumentParser()

# Add the arguments for the train and test folders
parser.add_argument('--train_images', help='Path to the training images dataset')
parser.add_argument('--train_annotations', help='Path to the training annotations dataset')
parser.add_argument('--val_images', help='Path to the validation images dataset')
parser.add_argument('--val_annotations', help='Path to the validation annotations dataset')
parser.add_argument('--model_path', help='Path to the model weights')


# Parse the command-line arguments
args = parser.parse_args()


if (not args.train_images) or (not args.train_annotations) or (not args.val_images) or (not args.val_annotations) :
    print("Expected command line arguments not found so exiting the program")

train_images = args.train_images
train_annotations = args.train_annotations
val_images = args.val_images
val_annotations = args.val_annotations
model_path = args.model_path

testdeeplabv3plus( train_images, train_annotations, val_images, val_annotations,model_path)

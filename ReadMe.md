# Transformavericks - Video Frame Prediction + Segmentation

# Instructions to reproduce our results

1) This repo has two main models , one is Video Frame Prediction and Segmentation
2) First task is to train video frame Prediction model
3) To do that first navigate to "VideoPred" Folder
4) The instructions to setup the model and generate future frames is mentioned in [VideoPred/README.md](VideoPred/README.md)
5) Once the future frames for validation and hidden sets are generated using the code and instructions mentioned in [VideoPred/README.md](VideoPred/README.md), Next task is to do segmentation on predicted frames and calcualte Jaccardindex for validationset and generate future segmentation frames for hidden set
6) The code and instructions to do that is clearly documented in  [Segmentation_Code/Readme.Md](Segmentation_Code/Readme.Md) which is present in "Segmentation_Code" folder
7) Following the instructions in Readme of Segmentation folder , you can generate the segmentation masks for both validation and hidden sets and calculate JaccardIndex for the hidden set.

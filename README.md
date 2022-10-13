# StarWarp:Lens-distortion-correction-based-on-StarGAN
This repository is the outcome of final year project: StarWarp. This project is a implementation of paper 'The GAN that Warped: Semantic Attribute Editing with Unpaired Data'. It modifies the StarGAN to translate images to warpfields. Warpfields keeps most of the detail of original images. It also could be scaled to apply different degree of editting, or applied on higher resolution.

The effect of this project is still limited due to lack of datasets, time, and computing resources.
![image](https://user-images.githubusercontent.com/83911295/195697536-01670e85-7599-4c3d-aacc-2004bf9b6127.png)


# Overview
![image](https://user-images.githubusercontent.com/83911295/195693867-93dc6788-7a35-4e5c-9e4d-606564eae9d7.png)

![image](https://user-images.githubusercontent.com/83911295/195694200-81715408-9a00-4675-ba50-7abd1408a04a.png)

![image](https://user-images.githubusercontent.com/83911295/195695164-77a1d43a-469f-45a2-997b-05c07c425448.png)


# Environment
python                    3.8.10            
pytorch(GPU)        1.9.0           
albumentations       1.1.0                   
numpy                    1.18.5                
tqdm                       4.62.4.dev6+g6c930f5     
pillow                     8.4.0
opencv                    4.0.1            
opencv-python             4.5.5.62                 
opencv-python-headless    4.5.5.62  

# Pretained weight
https://drive.google.com/drive/folders/1lWg_FDx5sDcZkCtbTbiW95itMgaI3kou?usp=sharing

# Reference and resource
[For more information see: [About](#about)](https://arxiv.org/abs/1811.12784)

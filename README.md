# StarWarp:Lens-distortion-correction-based-on-StarGAN
This repository is the outcome of final year project:StarWarp. In this project, after reviewing some of the mainstream research outcome in deep learning-based image translation, we constructed our lens distortion removing system based on deep learning method.

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

# Files
│── StarWarp8(root of project)
││── example1(example image for high resolution prediction)
││── example2 (example image for high resolution prediction)
││─ exmaple3(example image for high resolution prediction)
││─ log (log and pre-trained weight from epoch 5, 6, and 8)
││─ result (cache of training)
││─ cache (Temporary folder for TensorBoard)
│──d_checkpoint.tar(discriminator checkpoint)
│── g_checkpoint.tar(generator checkpoint)
│── main.py(Main python script with code of training)
│── config.py(Python script with hyper-parameters used in training)
│── models.py(Python script of Generator and discriminator)
│── utils.py(Python script of code of all related function include loss, warping function and color wheel drawing for visualization)


# Overview
![VUUM{D%`E@N$KI @~YZ XOM](https://user-images.githubusercontent.com/83911295/164816876-44411f40-832d-4adf-b716-cfa434e30eeb.jpg)
![image](https://user-images.githubusercontent.com/83911295/164816936-98b8f39a-1c4e-470f-be2f-ee3b5efc52bf.png)

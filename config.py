import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Path
ACTUAL_GLASSES_ROOT = "E:/finalyrs_project/real_imgs"
PASTED_GLASSES_ROOT = "E:/finalyrs_project/DatasetAugmentation/pasted"
MASK_ROOT = "E:/finalyrs_project/masks"
# REAL_MASK_ROOT = "E:/finalyrs_project/final_mask"
TRANSFORM = True

#Model
SAVE_MODEL = True
LOAD_MODEL = True
FLOAT16 = False

#Dataset
TEST_SPLIT = None#240
SHUFFLE = True
IMAGE_SIZE = 128 #144 as default
BATCH_SIZE = 16 #when float 16 on lab pc
NUM_WORKERS = 0
EPOCH = 1000

#Hyper parameters
LEARNING_RATE = 1e-4
CRITIC = 5
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_GP = 10
LAMBDA_CLS = 0.25
LAMBDA_CYCLE = 10
LAMBDA_SMOOTH = 125
LAMBDA_MASK = 1


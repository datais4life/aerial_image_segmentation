# The dataset used for this segmentation demonstration will consist of a subset of 200 aerial images and their masks from a larger dataset from the Massachusetts Roads Dataset. This larger dataset has a total of 1171 aerial images of the state of Massachusetts which cover 2.25 square kilometers in area. Each image and its corresponding mask is 1500x1500 pixels in size. The full dataset can me accessed at https://www.cs.toronto.edu/~vmnih/data/.
# Subset Dataset Download
# git clone https://github.com/parth1620/Road_seg_dataset.git

# Import Libraries Needed
import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn

from model import *

import sys
sys.path.append('/Road_seg_dataset')


# Define Variables
CSV_FILE = '/data/Aerial_Image_Segmentation/Road_seg_dataset/train.csv'
DATA_DIR = '/data/Aerial_Image_Segmentation/Road_seg_dataset/'

DEVICE = 'cuda'

EPOCHS = 15
LR = 0.003
BATCH_SIZE = 8
IMG_SIZE = 512

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

# Read the data file
df = pd.read_csv(CSV_FILE)

idx = 15

row = df.iloc[idx]

image_path = DATA_DIR + row.images
mask_path = DATA_DIR + row.masks

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/ 255

# Configure the formatting of the images and masks
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')

# Split the data into training and testing sections
train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

# Augmentation Functions
def get_train_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)
  ])

def get_valid_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE)
  ])

# Create a class to make a custom dataset for training images and the validated masks.
class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]

    image_path = DATA_DIR + row.images
    mask_path = DATA_DIR + row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #(h, w, c)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h, w)
    mask = np.expand_dims(mask, axis = -1) #(h, w, c)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image'] #(h, w, c)
      mask = data['mask']

    image = np.transpose(image, (2, 0, 1)).astype(np.float32) #(c, h, w)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32) #(c, h, w)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image, mask

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

# Loading the dataset into batches
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

model = SegmentationModel()
model.to(DEVICE);

# Training Function
def train_fn(dataloader, model, optimizer):

  model.train() # Turn ON dropout, batchnorm, etc..

  total_loss = 0.0

  for images, masks in tqdm(dataloader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(dataloader)

# Validation Function
def eval_fn(dataloader, model):

  model.eval() # Turn OFF dropout, batchnorm, etc..

  total_loss = 0.0

  with torch.no_grad():

    for images, masks in tqdm(dataloader):

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)
      logits, loss = model(images, masks)
      total_loss += loss.item()

  return total_loss / len(dataloader)

# Model Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

# Model Training
best_loss = np.Inf

for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model)

  if valid_loss < best_loss:
    torch.save(model.state_dict(), "best-model.pt") # Saving model weights
    print("SAVED-MODEL")
    best_loss = valid_loss

  print(f"Epoch : {i+1} TRAIN LOSS : {train_loss} VALID LOSS : {valid_loss}")
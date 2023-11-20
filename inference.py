import torch
#import segmentation_models_pytorch as smp
import helper
from model import SegmentationModel
from torch import nn
import argparse
from PIL import Image
import os
import cv2

#create parser
parser = argparse.ArgumentParser()

# add directory path that needs to be passed through. 
parser.add_argument("--input_dir", help="directory storing the input image files.", type=str)
parser.add_argument("--output_dir", help="directory storing the output prediction masks .", type=str)

#parse arguments
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

# Running model inference on an image. Change the index number to run the model on another image. 
idx = 22

DEVICE = 'cuda'

model = SegmentationModel()
model.to(DEVICE);

print(f"Model was loaded")
print(f"Using GPU : {torch.cuda.is_available()}")

#validset = SegmentationDataset(valid_df, get_valid_augs())

model.load_state_dict(torch.load('/data/Aerial_Image_Segmentation/best-model.pt'))
#image, mask = validset[idx]



# loop through folder full of images
def predict_on_folder_of_images(input_dir, output_dir):
    """
        SUMMARY
            predicts where a road would be using a Unet model weights to output a binary segmentation mask for the roads.
        ARGS
            input_dir(str): the directory/path where the input images are for prediction 
            output_dir(str): the directory/path where the output images are for prediction 
        RETURNS: NONE
    """
    
    # reading image folder
    image_count = 0
    print(f"reading image folder:{input_dir}")
    for file in os.listdir(input_dir):
        print(f"reading image: {file}")
        #image = Image.open(f"{input_dir}/{file}")
        
        image = cv2.imread(/data/Aerial_Image_Segmentation/Road_seg_dataset/images)
        image = cv2.cvtColor(input_dir, cv2.COLOR_BGR2RGB) #(h, w, c)
        print(f"reading image with cv2 : {file}")

        logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h, w) -> (b, c, h, w)
        pred_mask = torch.sigmoid(logits_mask)
        # filter above 50 percent
        pred_mask = (pred_mask > 0.5)*1.0

        # save prediction mask as an image
        pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert to uint8 format
        cv2.imwrite(f'{output_dir}/{file}_mask.png', pred_mask)
        image_count += 1 

# logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h, w) -> (b, c, h, w)
# pred_mask = torch.sigmoid(logits_mask)
# pred_mask = (pred_mask > 0.5)*1.0

#helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))

if __name__ == '__main__':
    try:
        predict_on_folder_of_images(input_dir, output_dir)
    except Exception as e:
        print(e)

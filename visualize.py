import torchvision
import os
from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
from pytorch_lightning import Trainer
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json


from classes import *

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
path = f'{img_folder}/test/images'
test_set = CocoDetection(img_folder=path, feature_extractor=feature_extractor, mode='test')

bigDict = {}
bigDict['x'] = []
bigDict['y'] = []
bigDict['z'] = []
bigDict['q'] = []

files = os.listdir("test/images")
files = sorted(files)

idList = {"images":[]}
idCount = 0
for f in files:
    if(f[-4:]=='json'):
        files.remove(f)
    temptDict = {"file_name":f,"id":idCount}
    idList["images"].append(temptDict)
    idCount+=1

with open(f'{img_folder}/test/images/custom_test.json', 'w') as f:
    json.dump(idList, f)


model.load_state_dict(torch.load('model.pth'))

device = torch.device("cuda")
model.to(device)

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    #extracting x and y

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def predict(i):

    xy = []
    z = []
    quality = []

    def plot_results(pil_img, prob, boxes):
        draw = ImageDraw.Draw(pil_img)
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = "cell"
            xy.append((('{:.4f}'.format((xmax-xmin)/2  + xmin)), '{:.4f}'.format( ( (ymax-ymin)/2 + ymin)))) 

            draw.text((xmin, ymin), text, fill='red')
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='blue', alpha=0.5))
        # count.append(len(prob))
        plt.axis('off')
        plt.savefig(f"test/pred_000{image.filename[-7:-4]}.png")

    def visualize_predictions(image, outputs, threshold=0.9):
        # keep only predictions with confidence >= threshold
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        confidence = probas[keep].max(-1).values
        quality.append(confidence.cpu().detach().numpy())
        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
        # plot results
        plot_results(image, probas[keep], bboxes_scaled)

    pixel_values, target = test_set[i]

    pixel_values = pixel_values.unsqueeze(0).to(device)

    # # forward pass to get class logits and bounding boxes
    outputs = model(pixel_values=pixel_values, pixel_mask=None)

    image_id = target['image_id'].item()

    image = test_set.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(f'{img_folder}/test/images', image['file_name']))

    # #WORKS ON 1 PICTURE. MODIFY TO VIEW OTHER PICTURES
    visualize_predictions(image, outputs)

    img_w, img_h = image.size

    for f in range(0,len(xy)):
        bigDict['x'].append(float(xy[f][0]))
        bigDict['y'].append(float(xy[f][1]))
        bigDict['z'].append(int(image.filename[-7:-4]))
        bigDict['q'].append(float(quality[0][f]))

for i in range(0,len(files)-1):
    predict(i)

    # Introduction
    # Methods
        #Running on 2080
    # results
    # Discussion
    #20 pages
    #talk about dbscan
df = pd.DataFrame(bigDict)
print(df.to_string(index=False))
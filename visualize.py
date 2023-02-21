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


from classes import *

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

model.load_state_dict(torch.load('model.pth'))

device = torch.device("cuda")
model.to(device)

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

xList = []
yList = []
quality = []

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    #extracting x and y
    # print('x ----------------------')
    # print(x_c.cpu().detach().numpy())
    # print('y ----------------------')
    # print(y_c.cpu().detach().numpy())
    xList.append(x_c.cpu().detach().numpy())
    yList.append(y_c.cpu().detach().numpy())

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = "cell"
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    plt.savefig("cell.png")

def visualize_predictions(image, outputs, threshold=0.9):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  confidence = probas[keep].max(-1).values
#   print('confidence ----------------------')
#   print(confidence.cpu().detach().numpy())
  quality.append(confidence.cpu().detach().numpy())
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
  # plot results
  plot_results(image, probas[keep], bboxes_scaled)

  #We can use the image_id in target to know which image it is



#Each file being tested on needs to be in a json list

pixel_values, target = val_dataset[13]

pixel_values = pixel_values.unsqueeze(0).to(device)
#print(pixel_values.shape)

# # forward pass to get class logits and bounding boxes
outputs = model(pixel_values=pixel_values, pixel_mask=None)

image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(f'{img_folder}/val/images', image['file_name']))
# #WORKS ON 1 PICTURE. MODIFY TO VIEW OTHER PICTURES
visualize_predictions(image, outputs)

for i in range(0,len(xList)):
#     print(str(xList[i]) + ' ' + str(yList[i]) + ' ' + str(quality[i]))

    print(xList[i])
    print(yList[i])
    print(quality[i])
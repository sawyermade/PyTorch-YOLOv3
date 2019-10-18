from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import pyrealsense2 as rs, cv2, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
# print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

# dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
#                         batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

# Realsense 2 setup
width, height = 640, 480
img_shape = (opt.img_size, opt.img_size)
dim_diff = np.abs(width - height)
pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
pad = ((pad1, pad2), (0, 0), (0, 0)) if height <= width else ((0, 0), (pad1, pad2), (0, 0))
# The amount of padding that was added
pad_x = max(height - width, 0) * (opt.img_size / max(height, width))
pad_y = max(width - height, 0) * (opt.img_size / max(height, width))
# Image height and width after padding is removed
unpad_h = opt.img_size - pad_y
unpad_w = opt.img_size - pad_x

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
profile = pipeline.start(config)

print ('\nPerforming object detection:')
startup_count = 0
while True:
    # Get frames
    frames = pipeline.wait_for_frames()
    frame_rgb = np.asanyarray(frames.get_color_frame().get_data())
    
    if startup_count < 20:
        startup_count += 1
        continue

    # Sends to detectron
    # retList = upload(url, frame)
    # if not retList:
    #     continue

    # Shows img
    # visImg = retList[0]
    # visImg = cv2.resize(visImg, (1200, 900))

    # Converts image and runs inference

    # input_img = torch.from_numpy(frame_rgb)
    # input_img.unsqueeze_(0)
    # print(input_img.shape)
    # input_img = Variable(input_img.type(Tensor).float())
    # print(input_img.shape)

    input_img = np.pad(frame_rgb, pad, 'constant', constant_values=127.5) / 255.
    input_img = resize(input_img, (*img_shape, 3), mode='reflect')
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = torch.from_numpy(input_img).float()
    input_img.unsqueeze_(0)
    input_img = Variable(input_img.type(Tensor))
    detections = None
    with torch.no_grad():
        # Gets detections
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
        detections = detections[0]

    # Draw bounding boxes and labels of detections
    img_bbox = np.copy(frame_rgb)
    if detections is not None:
        print('\tFrame:')
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * height
            box_w = ((x2 - x1) / unpad_w) * width
            y1b = ((y1 - pad_y // 2) / unpad_h) * height
            x1b = ((x1 - pad_x // 2) / unpad_w) * width
            y2b = y1b + box_h 
            x2b = x1b + box_w

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = [int(x * 255) for x in color]
            color = tuple(color[:-1])
            # print(color)
            # Create a Rectangle patch
            # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
            #                         edgecolor=color,
            #                         facecolor='none')

            # print(f'point1 = ({x1}, {y1}), point2 = ({x2}, {y2})')
            # input_img_og = np.pad(frame_rgb, pad, 'constant', constant_values=128)
            # print(f'input_img_og.shape = {input_img_og.shape}')
            # input_img_resize = resize(input_img_og, (*img_shape, 3), mode='reflect')
            # input_img_resize = cv2.resize(input_img_og, (opt.img_size, opt.img_size))
            # img_bbox = cv2.rectangle(input_img_resize, (x1, y1), (x2, y2), color)
            img_bbox = cv2.rectangle(img_bbox, (x1b, y1b), (x2b, y2b), color)
        # Views image
        cv2.imshow('Inference', cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == 1:
            continue

            # # Add the bbox to the plot
            # ax.add_patch(bbox)
            # # Add label
            # plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
            #         bbox={'color': color, 'pad': 0})
      

    # # Views image
    # cv2.imshow('Inference', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    #     break
    # elif k == 1:
    #     continue


# prev_time = time.time()
# for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
#     # Configure input
#     input_imgs = Variable(input_imgs.type(Tensor))

#     # Get detections
#     with torch.no_grad():
#         detections = model(input_imgs)
#         detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


#     # Log progress
#     current_time = time.time()
#     inference_time = datetime.timedelta(seconds=current_time - prev_time)
#     prev_time = current_time
#     print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

#     # Save image and detections
#     imgs.extend(img_paths)
#     img_detections.extend(detections)

# Bounding-box colors
# cmap = plt.get_cmap('tab20b')
# colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# print ('\nSaving images:')
# # Iterate through images and save plot of detections
# for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

#     print ("(%d) Image: '%s'" % (img_i, path))

#     # Create plot
#     img = np.array(Image.open(path))
#     plt.figure()
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)

#   # The amount of padding that was added
#   pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
#   pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
#   # Image height and width after padding is removed
#   unpad_h = opt.img_size - pad_y
#   unpad_w = opt.img_size - pad_x

#     # Draw bounding boxes and labels of detections
#     if detections is not None:
#         unique_labels = detections[:, -1].cpu().unique()
#         n_cls_preds = len(unique_labels)
#         bbox_colors = random.sample(colors, n_cls_preds)
#         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

#             print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

#             # Rescale coordinates to original dimensions
#             box_h = ((y2 - y1) / unpad_h) * img.shape[0]
#             box_w = ((x2 - x1) / unpad_w) * img.shape[1]
#             y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
#             x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

#             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#             # Create a Rectangle patch
#             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
#                                     edgecolor=color,
#                                     facecolor='none')
#             # Add the bbox to the plot
#             ax.add_patch(bbox)
#             # Add label
#             plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
#                     bbox={'color': color, 'pad': 0})

#     # Save generated image with detections
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(NullLocator())
#     plt.gca().yaxis.set_major_locator(NullLocator())
#     plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
#     plt.close()

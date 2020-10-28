import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from ssd_indirect_direct import build_ssd
import argparse

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=512, type=int, help='SSD300 or SSD512')
parser.add_argument('--input_size_2', default=56, type=int, help='input size of the second network')
parser.add_argument('--expand_num', default=3, type=int, help='expand ratio around the license plate')
args = parser.parse_args()

if not os.path.isdir("./results"):
    os.mkdir("./results")

net = build_ssd('test', args.input_size, args.input_size_2, 2, args.expand_num)    # initialize SSD

# --------------------detection start---------------------------

net.load_weights("./weights/indirect_direct.pth")

# matplotlib inline
from matplotlib import pyplot as plt
from data import INDIRECT_DIRECTDetection, INDIRECT_DIRECTAnnotationTransform
INDIRECT_DIRECT_ROOT = "./images/"
testset = INDIRECT_DIRECTDetection(INDIRECT_DIRECT_ROOT, None, None, INDIRECT_DIRECTAnnotationTransform(),
                                       dataset_name='test')
for img_id in range(3):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    detections, detections_indirect, detections_direct = net(xx, [])

    from data import INDIRECT_DIRECT_CLASSES as labels

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_4 = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

# --------------------indirect & direct---------------------------

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = detections.data
    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        th = 0.5
        for j in range(detections.size(2)):
            if detections[0, i, j, 0] > th:
                pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1

                if i == 1:
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='#ffc000', linewidth=2))
                
                if i == 2:
                    lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                    lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
                    four_corners = (detections[0, i, j, 5:]*scale_4).cpu().numpy()
                    corners_x = np.append(four_corners[0::2], four_corners[0])
                    corners_y = np.append(four_corners[1::2], four_corners[1])
                    currentAxis.plot(corners_x, corners_y, linewidth=2, color='#ff0000')

    plt.savefig(os.path.join("./results", str(img_id)+"_indirect_direct"+".svg"), bbox_inches='tight')
    plt.close()

# --------------------indirect---------------------------

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections_indirect = detections_indirect.data
    for i in range(detections_indirect.size(1)):
        # skip background
        if i == 0:
            continue
        th = 0.5
        for j in range(detections_indirect.size(2)):
            if detections_indirect[0, i, j, 0] > th:
                pt = (detections_indirect[0, i, j, 1:5]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1

                if i == 1:
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='#ffc000', linewidth=2))
                
                if i == 2:
                    lp_pt = (detections_indirect[0, i, j, 1:5]*scale).cpu().numpy()
                    lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
                    four_corners = (detections_indirect[0, i, j, 5:]*scale_4).cpu().numpy()
                    corners_x = np.append(four_corners[0::2], four_corners[0])
                    corners_y = np.append(four_corners[1::2], four_corners[1])
                    currentAxis.plot(corners_x, corners_y, linewidth=2, color='#ff0000')

    plt.savefig(os.path.join("./results", str(img_id)+"_indirect"+".svg"), bbox_inches='tight')
    plt.close()

# --------------------direct---------------------------

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections_direct = detections_direct.data
    for i in range(detections_direct.size(1)):
        # skip background
        if i == 0:
            continue
        th = 0.5
        for j in range(detections_direct.size(2)):
            if detections_direct[0, i, j, 0] > th:
                pt = (detections_direct[0, i, j, 1:5]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                
                if i == 1:
                    lp_pt = (detections_direct[0, i, j, 1:5]*scale).cpu().numpy()
                    lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
                    four_corners = (detections_direct[0, i, j, 5:]*scale_4).cpu().numpy()
                    corners_x = np.append(four_corners[0::2], four_corners[0])
                    corners_y = np.append(four_corners[1::2], four_corners[1])
                    currentAxis.plot(corners_x, corners_y, linewidth=2, color='#ff0000')

    plt.savefig(os.path.join("./results", str(img_id)+"_direct"+".svg"), bbox_inches='tight')
    plt.close()


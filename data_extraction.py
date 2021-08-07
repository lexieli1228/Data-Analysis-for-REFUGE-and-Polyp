import matplotlib
import os
from os import listdir
from os.path import isdir, join
import re
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import cv2
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import json
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import warnings
import pdb

def read_x():
    list_file_path = "orginal file path"
    with open(list_file_path, 'r') as f:
        training_details = f.readlines()
        epoch_np = np.array(0)
        loss_num_f_np = np.array(0)
        ce_num_f_np = np.array(0)
        dice_num_np = np.array(0)
        for i in range(len(training_details)):
            if i == 0 or i == 1:
                continue
            if i == (len(training_details) - 2):
                continue
            if training_details[i][15] == 's':
                continue
            temp_0 = training_details[i]
            temp_1 = temp_0.replace('\n', '')
            temp_2 = temp_1.replace(' ', '_')
            time_head, epoch_num, loss_x, loss_num, ce_x, ce_num, dice_x, dice_num = temp_2.split('_')
            loss_num_f = loss_num.replace(',', '')
            ce_num_f = ce_num.replace(',', '')
            if int(epoch_num) > 499:
                break
            epoch_np = np.append(epoch_np, int(epoch_num))
            loss_num_f_np = np.append(loss_num_f_np, float(loss_num_f))
            ce_num_f_np = np.append(ce_num_f_np, float(ce_num_f))
            dice_num_np = np.append(dice_num_np, float(dice_num))

        np.savetxt("log_polyp_dunet_loss.txt", loss_num_f_np, fmt='%f', delimiter=',')
        np.savetxt("log_polyp_dunet_ce.txt", ce_num_f_np, fmt='%f', delimiter=',')
        np.savetxt("log_polyp_dunet_dice.txt", dice_num_np, fmt='%f', delimiter=',')


if __name__ == '__main__':
    read_x()
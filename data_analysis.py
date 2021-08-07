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
    # refuge dataset
    # loss
    loss_num_f_np_refuge_segtran = np.loadtxt('log_refuge_segtran_loss.txt', delimiter=',')
    loss_num_f_np_refuge_unet = np.loadtxt('log_refuge_unet_loss.txt', delimiter=',')
    loss_num_f_np_refuge_pranet = np.loadtxt('log_refuge_pranet_loss.txt', delimiter=',')
    loss_num_f_np_refuge_dunet = np.loadtxt('log_refuge_dunet_loss.txt', delimiter=',')

    # ce
    ce_num_f_np_refuge_segtran = np.loadtxt('log_refuge_segtran_ce.txt', delimiter=',')
    ce_num_f_np_refuge_unet = np.loadtxt('/log_refuge_unet_ce.txt', delimiter=',')
    ce_num_f_np_refuge_pranet = np.loadtxt('log_refuge_pranet_ce.txt', delimiter=',')
    ce_num_f_np_refuge_dunet = np.loadtxt('log_refuge_dunet_ce.txt', delimiter=',')

    # dice
    dice_num_f_np_refuge_segtran = np.loadtxt('log_refuge_segtran_dice.txt', delimiter=',')
    dice_num_f_np_refuge_unet = np.loadtxt('log_refuge_unet_dice.txt', delimiter=',')
    dice_num_f_np_refuge_pranet = np.loadtxt('log_refuge_pranet_dice.txt', delimiter=',')
    dice_num_f_np_refuge_dunet = np.loadtxt('log_refuge_dunet_dice.txt', delimiter=',')

    # polyp dataset
    # loss
    loss_num_f_np_polyp_segtran = np.loadtxt('log_polyp_segtran_loss.txt', delimiter=',')
    loss_num_f_np_polyp_unet = np.loadtxt('log_polyp_unet_loss.txt', delimiter=',')
    loss_num_f_np_polyp_pranet = np.loadtxt('log_polyp_pranet_loss.txt', delimiter=',')
    loss_num_f_np_polyp_dunet = np.loadtxt('log_polyp_dunet_loss.txt', delimiter=',')

    # ce
    ce_num_f_np_polyp_segtran = np.loadtxt('log_polyp_segtran_ce.txt', delimiter=',')
    ce_num_f_np_polyp_unet = np.loadtxt('log_polyp_unet_ce.txt', delimiter=',')
    ce_num_f_np_polyp_pranet = np.loadtxt('log_polyp_pranet_ce.txt', delimiter=',')
    ce_num_f_np_polyp_dunet = np.loadtxt('log_polyp_dunet_ce.txt', delimiter=',')

    # dice
    dice_num_f_np_polyp_segtran = np.loadtxt('log_polyp_segtran_dice.txt', delimiter=',')
    dice_num_f_np_polyp_unet = np.loadtxt('log_polyp_unet_dice.txt', delimiter=',')
    dice_num_f_np_polyp_pranet = np.loadtxt('log_polyp_pranet_dice.txt', delimiter=',')
    dice_num_f_np_polyp_dunet = np.loadtxt('D:log_polyp_dunet_dice.txt', delimiter=',')

    names = range(0, 500)
    names = [str(x) for x in list(names)]
    x = range(len(names))
    # dice: refuge--salmon, polyp--orchid
    # ce: refuge--teal, polyp--yellowgreen
    # loss: refuge--orange, polyp--coral
    # refuge: segtran--orchid , unet--teal  , pranet--coral, dunet--turquoise
    # polyp: segtran--orchid , unet--teal , pranet--coral , dunet--turquoise
    plt.plot(x, loss_num_f_np_polyp_segtran, color='orchid', mec='r', mfc='w', label='segtran')
    plt.plot(x, loss_num_f_np_polyp_unet, color='teal', mec='r', mfc='w', label='unet')
    plt.plot(x, loss_num_f_np_polyp_pranet, color='coral', mec='r', mfc='w', label='pranet')
    plt.plot(x, loss_num_f_np_polyp_dunet, color='turquoise', mec='r', mfc='w', label='dunet')
    plt.legend()
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('epoch')
    plt.ylabel('polyp: loss')
    plt.show()

if __name__ == '__main__':
    read_x()
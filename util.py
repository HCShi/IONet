import argparse
import numpy as np
import torch
import json
import subprocess
import pandas as pd
import matlab.engine
from scipy.interpolate import interp1d
from skimage.morphology import dilation
from scipy.signal import medfilt
from skimage import measure
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
#import matplotlib.pyplot as plt

def str2bool(v):
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
         return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
         return False
     else:
         raise argparse.ArgumentTypeError('Unsupported value encountered.')
def get_args():
    parser = argparse.ArgumentParser()
    # ================== about load pre model   ################################
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--load_model_name', type=str, default='')
    # ================== about generator module ################################
    parser.add_argument('--multi_class', type=str2bool, default=True)
    parser.add_argument('--video_feature_dim', type=int, default=1024,  #1024
                        help='the dim of features')
    parser.add_argument('--image_feature_dim', type=int, default=1024,
                        help='the dim of features')
    parser.add_argument('--Vu_middle_feature_dim', type=int, default=1024,
                        help='the dim of features')
    parser.add_argument('--Vt_middle_feature_dim', type=int, default=1024,  #1024
                        help='the dim of features')
    parser.add_argument('--DV_middle_feature_dim', type=int, default=512,  #256
                        help='the dim of features')

    # ================== about classifier module ################################
    parser.add_argument('--f_middle1_feature_dim', type=int, default=1024,
                        help='the dim of features')
    parser.add_argument('--f_middle2_feature_dim', type=int, default=512,
                        help='the dim of features')
    parser.add_argument('--f_class_dim', type=int, default=21,
                        help='the dim of features')

    # ================== about gan module ################################
    parser.add_argument('--gan_mode', type=str, default='',
                        help='the loss type of gan')
    parser.add_argument('--lambda_bg', type=float, default=.1) #0.1
    parser.add_argument('--lambda_att', type=float, default=.1) #0.1

    # ================= about train ############################################
    parser.add_argument('--isTrain', type=bool, default= True, help= 'train/test')
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--lr_steps', default=[40, 80, 120, 500], type=float, nargs="+", metavar='LRSteps')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--class_num', type=int, default=20)
    parser.add_argument('--optimization_strategy', type=str, default='SGD')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)

    # ================= print setting ==========================================
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--print_freq_val', type=int, default=210)
    parser.add_argument('--eval_freq', type=int, default=5)
    # ================= coefficient  ===========================================
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.2)
    # ================= optial flow ============================================
    parser.add_argument('--flow_mode', type= str2bool, default=False)
    parser.add_argument('--attention_mode', type= str2bool, default=True)
    parser.add_argument('--frozen_feature_net', type= str2bool, default=True)

    # ================= about dataset ==========================================
    parser.add_argument('--train_video_txt', type=str, default='./thumos14_train_list.txt')
    parser.add_argument('--val_video_txt', type=str, default='./thumos14_test_list.txt')
    parser.add_argument('--class_txt', type=str, default='./thumos14_class.txt')
    parser.add_argument('--image_txt', type=str, default='./UCF101_list.txt')
    parser.add_argument('--image_root_dir', type=str, default='ucf')
    parser.add_argument('--interval', type=int, default=30, help='the rate of sampling')
    # ================= about detect ============================================
    parser.add_argument('--thrh', type=int, default=1)
    parser.add_argument('--pro', type=int, default=1)
    parser.add_argument('--weight_global', type=float, default=0.25)
    parser.add_argument('--sample_offset', type=float, default=0.0)
    parser.add_argument('--global_score_thrh', type=float, default=0.1)
    parser.add_argument('--fps', type=float, default=30.0)
    return parser.parse_args()

##################fun##########################
def normalize(x):
    x -= x.min()
    x /= x.max()
    return x

def interpolate(x, frame_cnt, sample_rate, snippet_size=None, kind='linear'):
    '''Upsample the sequence the original video fps.'''

    frame_ticks = __get_frame_ticks(frame_cnt, sample_rate,snippet_size)

    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1] + 1)
    # frame_ticks[-1] included

    interp_func = interp1d(frame_ticks, x, kind=kind)
    out = interp_func(full_ticks)

    return out

def __get_frame_ticks(frame_cnt, sample_rate, snippet_size=None):
    '''Get the frames of each feature snippet location.'''
    assert (snippet_size is not None)
    clipped_length = frame_cnt - snippet_size
    clipped_length = (clipped_length // sample_rate) * sample_rate
    # the start of the last chunk

    frame_ticks = np.arange(0, clipped_length + 1, sample_rate)
    # From 0, the start of chunks, clipped_length included
    return frame_ticks

def detect_with_thresholding(metric, thrh, pro):
    mask = metric > thrh         #max thread
    mask = medfilt(mask[:, 0], kernel_size=pro)  #median
    # kernel_size should be odd
    mask = np.expand_dims(mask, axis=1)
    return mask

### for optimizing the detection results
def mask_to_detections(mask, metric):
    out_detections = []
    detection_map = measure.label(mask, background=0)
    detection_num = detection_map.max()

    for detection_id in range(1, detection_num + 1):

        start = np.where(detection_map == detection_id)[0].min()   #the start
        end = np.where(detection_map == detection_id)[0].max() + 1  # the end
        start_area = metric[detection_map == detection_id]
        left = metric[start:start, :]
        right = metric[end:end, :]

        end_area = np.concatenate((left, right), axis=0)

        if end_area.shape[0] == 0:
            detection_score = start_area.mean() * 1
        else:
            detection_score = (start_area.mean() * 1 + end_area.mean() * (-1))

        out_detections.append([start, end, None, detection_score])   #start and end, confidence
    return out_detections

matlab_eng = matlab.engine.start_matlab()

thumos14_old_cls_names = { 7: 'BaseballPitch', 9: 'BasketballDunk', 12: 'Billiards', 21: 'CleanAndJerk', 22: 'CliffDiving',
    23: 'CricketBowling', 24: 'CricketShot', 26: 'Diving', 31: 'FrisbeeCatch', 33: 'GolfSwing', 36: 'HammerThrow', 40: 'HighJump',
    45: 'JavelinThrow', 51: 'LongJump', 68: 'PoleVault',  79: 'Shotput',  85: 'SoccerPenalty', 92: 'TennisSwing', 93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}

thumos14_old_cls_indices = {v: k for k, v in thumos14_old_cls_names.items()}

thumos14_new_cls_names = { 0: 'BaseballPitch', 1: 'BasketballDunk', 2: 'Billiards', 3: 'CleanAndJerk', 4: 'CliffDiving',
    5: 'CricketBowling', 6: 'CricketShot', 7: 'Diving', 8: 'FrisbeeCatch', 9: 'GolfSwing', 10: 'HammerThrow', 11: 'HighJump',
    12: 'JavelinThrow', 13: 'LongJump', 14: 'PoleVault', 15: 'Shotput', 16: 'SoccerPenalty', 17: 'TennisSwing', 18: 'ThrowDiscus',
    19: 'VolleyballSpiking', 20: 'Background',
}

thumos14_new_cls_indices = {v: k for k, v in thumos14_new_cls_names.items()}

old_cls_names = {'thumos14': thumos14_old_cls_names,}

old_cls_indices = {'thumos14': thumos14_old_cls_indices,}

new_cls_names = {'thumos14': thumos14_new_cls_names,}

new_cls_indices = {'thumos14': thumos14_new_cls_indices,}

def eval_thumos_detect(detfilename, gtpath, subset, threshold):
    assert (subset in ['test', 'val'])
    matlab_eng.addpath('THUMOS14_evalkit_20150930')
    aps = matlab_eng.TH14evalDet(detfilename, gtpath, subset, threshold)
    aps = np.array(aps)
    mean_ap = aps.mean()
    return aps, mean_ap

def output_detections_thumos14(out_detections, out_file_name):

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names['thumos14'][class_id]
        old_class_id = int(old_cls_indices['thumos14'][class_name])
        entry[3] = old_class_id

    out_file = open(out_file_name, 'w')

    for entry in out_detections:
        out_file.write('{} {:.2f} {:.2f} {} {:.4f}\n'.format(
            entry[0], entry[1], entry[2], int(entry[3]), entry[4]))

    out_file.close()
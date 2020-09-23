from torch.utils.data import Dataset
from util import get_args
import numpy as np
import torch
import imageio
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os

class VideoDataset(Dataset):
    """Video dataset.(There is the THUMOS 14.)"""
    def __init__(self, args, transform = None, test_mode= True):

        self._test_mode = test_mode     # default=train

        self._class_txt = args.class_txt
        self._action_class = self._get_action_class()  # total action class
        self._num_class = len(self._action_class)  # class num

        self._class_to_ind = dict(
            list(zip(self._action_class, list(range(self._num_class)))))  # zip:将每条action_class和总的class打包

        self._flow_mode = args.flow_mode  # default=False

        if self._test_mode:
            self._root_dir = args.video_val_root_dir     #val_root
            self._video_txt = args.val_video_txt    #val_video_txt  /thumos14_test_list
        else:
            self._root_dir = args.video_train_root_dir    #train_root
            self._video_txt = args.train_video_txt   #train_video_txt  /thumos14_train_list

        self._video_list, self._video_label = self._get_video_list_label()  # video_list, video_label
        self._num_video = len(self._video_list)  # video_num

        self._transform = transform

    def _get_action_class(self):
        with open(self._class_txt) as f:
            action_class = [line.strip() for line in f.readlines()]
        return action_class

    def _get_video_list_label(self):
        video_list = []
        video_label = {}
        with open(self._video_txt) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                _ = line.strip().split("  ")
                video_list.append(_[0])
                video_label[_[0]] = [int(i) for i in _[1].split()]
        return video_list, video_label

    def __len__(self):
        """ return the number of train and test data """
        return self._num_video

    def __getitem__(self, index):
        video_name = self._video_list[index]   #each video

        if self._flow_mode:
            path = self._root_dir + '/' + video_name + '.npy'
            video_path = np.load(path)
            #video_feature = torch.squeeze(torch.from_numpy(video_path['feature']).float(), dim=0)
            video= torch.from_numpy(video_path).float()
            video_label_hot = torch.tensor(self._video_label[video_name])
            video_label = torch.squeeze(torch.nonzero(video_label_hot))

        else:
            path = self._root_dir + '/' + video_name + '.npy'
            video_path = np.load(path)
            # video_feature = torch.squeeze(torch.from_numpy(video_path['feature']).float(), dim=0)
            video = torch.from_numpy(video_path).float()

            video_label_hot = torch.tensor(self._video_label[video_name])
            video_label = torch.squeeze(torch.nonzero(video_label_hot))
        return video, video_label, video_name

class TrainDataset(Dataset):
    def __init__(self, args, transform=None, test_mode=False):
        self._test_mode = test_mode  # default=train
        self._transform = transform

        self._video_txt = args.train_video_txt  #
        self._video_list, self._video_label = self._get_video_list_label()  # video_list, video_label
        self._num_video = len(self._video_list)  # video_num

        self._image_txt = args.image_txt  # UCF101_list
        self._image_list, self._image_label = self._get_image_list_label()  # image_list, image_label
        self._num_image = len(self._image_list)  # video_num

        self._flow_mode = args.flow_mode  # default=False

        self._video_root_dir = args.video_train_root_dir  # train_root
        self._image_root_dir = args.image_root_dir  # ucf_root

    def _get_video_list_label(self):
        video_list = []
        video_label = {}
        with open(self._video_txt) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                _ = line.strip().split("  ")
                video_list.append(_[0])
                video_label[_[0]] = [int(i) for i in _[1].split()]
        return video_list, video_label

    def _get_image_list_label(self):
        image_list = []
        image_label = {}
        with open(self._image_txt) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                _ = line.strip().split(" ")
                image_list.append(_[0])
                image_label[_[0]] = int(_[1])
        return image_list, image_label

    def __len__(self):
        """ return the number of train and test data """
        return self._num_video * self._num_image

    def __getitem__(self, index):

        video_index = int(index/self._num_image)
        video_name = self._video_list[video_index]   #each video

        if self._flow_mode:
            path = self._video_root_dir + '/' + video_name + '.npy'
            video_path = np.load(path)
            video_feature = torch.from_numpy(video_path).float()
            video_label_hot = torch.tensor(self._video_label[video_name])
            video_label = torch.squeeze(torch.nonzero(video_label_hot))

        else:
            path = self._video_root_dir + '/' + video_name + '.npy'

            video_path = np.load(path)
            video_feature = torch.from_numpy(video_path).float()

            video_label_hot = torch.tensor(self._video_label[video_name])
            video_label = torch.squeeze(torch.nonzero(video_label_hot))
        if video_label.dim() == 1:
            t = np.random.randint(0,2)
            video_label = video_label[t]

        return video_feature, video_label

if __name__ == '__main__':
    args = get_args()
    train_loader = torch.utils.data.DataLoader(
        TrainDataset(args=args,
                     transform=transforms.Compose([
                         transforms.CenterCrop((224,224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True)

    with torch.no_grad():
        for i, (image_feature, image_label, video_feature, video_label) in enumerate(train_loader):
            print(video_feature.size())
            print(video_label)




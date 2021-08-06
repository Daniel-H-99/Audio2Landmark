"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os.path
import numpy as np
import random
#import librosa
import pickle
from PIL import Image
import cv2
import utils


def path_extractor(name): #JK_a01_000.jpeg
    name = name.split('.')[0]
    character = name.split('_')[0]
    folder = name.split('_')[1]
    path = character + '/' + folder +'/'
    return path

def default_image_loader(path):
    return Image.open(path).convert('RGB')

def default_audioG_loader(path):
    fs = 44100
    y = librosa.load(path, fs)[0]
    fn = int(fs ** 2 / y.shape[0])
    y_down = librosa.resample(y, fs, fn)
    delta = fs - y_down.shape[0]
    for i in range(delta):
        y_down = np.append(y_down, y_down[-1*i])
    return y_down


def default_audioL_loader(path):
    fs = 44100
    return librosa.load(path, fs)[0]

def default_pickle_loader(path):
    with open(path, 'rb') as rf:
        data = pickle.load(rf)
    return data


def default_audiolist_reader(flist):
    sample_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            audio_gpath = line.strip().split(' ')[0]
            audio_local_sample = line.strip().split(' ')[1:]
            audio_local_path = path_extractor(audio_local_sample[0])
            audio_local_path_0 = audio_local_path + audio_local_sample[0]
            audio_local_path_1 = audio_local_path + audio_local_sample[1]
            audio_local_path_2 = audio_local_path + audio_local_sample[2]
            sample_list.append([audio_gpath, audio_local_path_0, audio_local_path_1, audio_local_path_2])

    return sample_list


def default_imagelist_reader(flist):
    sample_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            video_gpath = line.strip().split(' ')[0]
            video_label = line.strip().split(' ')[1]
            video_local_sample = line.strip().split(' ')[2:]
            video_local_path = path_extractor(video_local_sample[0])
            video_local_path_0 = video_local_path + video_local_sample[0]
            video_local_path_1 = video_local_path + video_local_sample[1]
            video_local_path_2 = video_local_path + video_local_sample[2]
            sample_list.append([video_gpath, video_label, video_local_path_0, video_local_path_1, video_local_path_2])

    return sample_list

def make_dataset(dir):
    data_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        data_paths.append(os.path.join(dir, fname))
    return data_paths

def default_picklelist_reader(flist):
    pickle_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            pickle = line.strip()
            pickle_list.append(pickle)
    return pickle_list

def default_config_reader(config):
    item_list = []
    with open(config, 'r') as rf:
        for line in rf.readlines():
            item = line.strip()
            item_list.append(item)
    return item_list

def default_parameter_reader(flist):
    parameter_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            parameters = line.strip().split(' ')[:-1]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
    return parameter_list

def default_comparameter_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.split(' ')[0]
            name_list.append(name)
            parameters = line.split(' ')[1:]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
    return name_list, parameter_list

def img_concat_tool(data1, data2, data3):
    input = torch.zeros([data1.shape[0], data1.shape[1], data1.shape[2], 3])
    input[:, :, :, 0] = data1
    input[:, :, :, 1] = data2
    input[:, :, :, 2] = data3
    return input

def aud_concat_tool(data1, data2, data3):
    input = np.append(data1, data2)
    input = np.append(input, data3)
    return input

def preprocess_tool(raw_audio, config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    peak = abs(max(abs(np.max(raw_audio)), abs(np.min(raw_audio))))
    raw_audio = raw_audio/peak
    raw_audio *= 256.0

    # Make minimum length available
    length = config
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))
        if length < raw_audio.shape[0]:
            raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, -1, 1])

    return raw_audio.copy()


class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """

    def __init__(self, phase, final_size):
        assert (phase == 'train' or phase == 'test')
        self.phase = phase
        self.final_size = final_size

    def __call__(self, img):
        # pre determined parameters
        final_size = self.final_size
        final_width = final_height = final_size
        res_img = img.resize((final_width, final_height))
        return res_img

class A2LDataset(data.Dataset):
    def __init__(self, config, split='train', audio_transform=None, img_transform=None):
        super(A2LDataset, self).__init__()
        self.split = split
        self.root = config['{}_root'.format(split)]
        self.config = default_config_reader(os.path.join(self.root, 'config.txt'))
        self.landmarks = default_pickle_loader(os.path.join(self.root, 'PCA_reducedKp.pickle'))
        self.audio_transform = audio_transform
        self.img_transform = img_transform
        self.rawKp = default_pickle_loader(os.path.join(self.root, 'rawKp.pickle'))
        self.seq_length = int(config['seq_length'])
        # vid, index = self.config[0].split('/')
        # N = 0
        # cnt = 0
        # for k in self.rawKp[vid].keys():
        #     N += self.rawKp[vid][k][1]
        #     cnt += 1
        # self.N = N / cnt

        print(len(self.config))
        
    def __getitem__(self, index):
        _index = index
        if not 'test' in self.split: 
            audios = []
            lms = []
            vid, index = self.config[_index].split('/')
            indice = [index]
            i = self.seq_length
            while len(indice) < self.seq_length:
                _index -= self.seq_length - i
                index = self.config[_index].split('/')[1]
                indice = [index]
                for i in range(1, self.seq_length):
                    if _index + i >= len(self.config):
                        break
                    tmp_vid, tmp_index = self.config[_index + i].split('/')
                    if vid != tmp_vid:
                        break
                    else:
                        indice.append(tmp_index)
                

            for index in indice:
                audio_path = os.path.join(self.root, vid, 'audio', '{:05d}'.format(int(index) - 1) + '.pickle')
                audio = default_pickle_loader(audio_path)
                if self.audio_transform is not None:
                    audio = self.audio_transform(audio)
                lm = self.landmarks[vid][index].reshape(-1)
                audios.append(audio)
                lms.append(lm)
        #         print(vid, index)
        #         print(audio.shape)

            audios = np.stack(audios, axis=0)
            lms = np.stack(lms, axis=0)
            # scale_coeff = utils.extract_scale_coeff(all_kp)
            # print("scale coeff: {}".format(scale_coeff))
            # return audio, np.concatenate([lm, scale_coeff], axis=0)
            return audios,lms 
        else:
            vid, index = self.config[_index].split('/')
            audio_path = os.path.join(self.root, vid, 'audio', '{:05d}'.format(int(index) - 1) + '.pickle')
            audio = default_pickle_loader(audio_path)
            if self.audio_transform is not None:
                audio = self.audio_transform(audio)
            lm = self.landmarks[vid][index].reshape(-1)
            _, N, theta, mean, _, all_kp = self.rawKp[vid][index]
            img_path = os.path.join(self.root, vid, 'img', index + '.png')
            img = cv2.imread(img_path)
            return audio, lm, N, theta, mean, all_kp, index, img
            
    def __len__(self):
        return len(self.config)
        
class A2BDataset(data.Dataset):
    def __init__(self, config, split='train', audio_transform=None, img_transform=None):
        super(A2BDataset, self).__init__()
        self.split = split
        self.root = config['{}_root'.format(split)] + '_b'
        self.config = default_config_reader(os.path.join(self.root, 'config.txt'))
        self.bfms = default_pickle_loader(os.path.join(self.root, 'bfms.pickle'))
        self.audio_transform = audio_transform
        self.img_transform = img_transform
        print(len(self.config))
        
    def __getitem__(self, index):
        vid, index = self.config[index].split('/')
        audio_path = os.path.join(self.root, vid, 'audio', index + '.pickle')
        audio = default_pickle_loader(audio_path)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
        if self.split == 'test':
            return audio
        
        bfm = self.bfms[vid][index].reshape(-1)
#         print(vid, index)
#         print(audio.shape)
        return audio, bfm

    def __len__(self):
        return len(self.config) * 30
        
class SMEDataset(data.Dataset):
    def __init__(self, root, flist, audio_transform = None, config = None, pickle_loader =default_pickle_loader,
                  picklelist_reader = default_picklelist_reader, paramter_reader = default_comparameter_reader):
        self.root = root

        self.audio_root = './mfcc'
        self.parameter_list = flist['parameter_list']
        self.audio_list = flist['audio_list']

        self.audio_transform = audio_transform
        self.config = config
        self.pickle_loader = pickle_loader
        self.picklelist_reader = picklelist_reader
        self.parameter_reader = paramter_reader

        self.audio_data = self.picklelist_reader(self.audio_list)
        self.ldmk_data = self.pickle_loader(self.parameter_list)

    def __getitem__(self, index):
        audio_name = self.audio_data[index]
        audio_path = os.path.join(self.audio_root, audio_name)
        audio = self.pickle_loader(audio_path)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        landmark = self.ldmk_data[index]
        landmark = torch.Tensor(landmark)

        return {'AU':audio, 'PM':landmark}

    def __len__(self):
        return len(self.audio_data) * 300


class SMEDtestset(data.Dataset):
    def __init__(self, root, flist, audio_transform = None, config = None, pickle_loader =default_pickle_loader,
                  picklelist_reader = default_picklelist_reader, paramter_reader = default_parameter_reader):
        self.root = root

        self.audio_root = './MFCC_test'
        self.parameter_list = flist['parameter_list_test']
        self.audio_list = flist['audio_list_test']

        self.audio_transform = audio_transform
        self.config = config
        self.pickle_loader = pickle_loader
        self.picklelist_reader = picklelist_reader
        self.parameter_reader = paramter_reader

        self.audio_data = self.picklelist_reader(self.audio_list)
        self.ldmk_data = self.parameter_reader(self.parameter_list)
        


    def __getitem__(self, index):
        audio_name = self.audio_data[index]
        image_name = self.audio_data[index].replace('pickle', 'jpg')
        image_name = image_name.replace('/', '_')
        audio_path = os.path.join(self.audio_root, audio_name)
        audio = self.pickle_loader(audio_path)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)


        return {'AU':audio, 'IN':image_name}

    def __len__(self):
        return len(self.audio_data) * 300


# class SaveeTestDataset(data.Dataset):
#     def __init__(self, root, flist, train, transform=None, audio_transform = None, preprocess=True, config = None, pickle_loader =default_pickle_loader,
#                 paramter_reader = default_comparameter_reader, picklelist_reader = default_picklelist_reader):
#         self.root = root
#         self.audio_root = os.path.join(self.root, 'Audio')
#         if train is not False:
#             self.parameter_list = flist['parameter_list']
#             self.phoneme_list = flist['phoneme_list']
#         else:
#             self.parameter_list = flist['parameter_list_test']
#             self.phoneme_list = flist['phoneme_list_test']
#
#         self.transform = transform
#         self.audio_transform = audio_transform
#         self.preprocess = preprocess
#         self.config = config
#
#         self.pickle_loader = pickle_loader
#         self.parameter_reader = paramter_reader
#         self.picklelist_reader = picklelist_reader
#
#         self.phoneme_data = self.picklelist_reader(self.phoneme_list)
#         self.parameter_name, self.parameter_data = self.parameter_reader(self.parameter_list)
#
#
#
#     def __getitem__(self, index):
#         phoneme_name = self.phoneme_data[index]
#         phoneme_path = os.path.join(self.audio_root, phoneme_name)
#         phoneme = self.pickle_loader(phoneme_path)
#         if self.audio_transform is not None:
#             phoneme = self.audio_transform(phoneme)
#
#         parameter_name = self.parameter_name[index]
#
#
#         return {'PH':phoneme,'PN':parameter_name} #
#
#     def __len__(self):
#         return len(self.phoneme_data)


# def get_data_loader_list(config, train):
#     transform_list = [transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))]
#     audio_transform = transforms.Compose(transform_list)
#     if train == True:
#         dataset = SMEDataset(config['root'], config['flist'],
#                                audio_transform=audio_transform, config=config['audio_load_size'])
#         loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=train, drop_last=True,
#                             num_workers=config['num_workers'])
#     else:
#         dataset = SMEDtestset(config['root'], config['flist'],
#                              audio_transform=audio_transform, config=config['audio_load_size'])
#         loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=train, drop_last=True,
#                             num_workers=config['num_workers'])
#     return loader

def get_data_loader_list(config, split='train', bfm=False):
    assert split in ['train', 'eval', 'test']
    batch_size = config['{}_batch_size'.format(split)]
    transform_list = [transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))]
    audio_transform = transforms.Compose(transform_list)
    
    dataset = A2BDataset(config, split,
                            audio_transform=audio_transform) if bfm else A2LDataset(config, split, audio_transform=audio_transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=split != 'test', drop_last=split != 'test',
                        num_workers=config['num_workers'])
    print("data size: {}".format(len(loader)))
    
    return loader

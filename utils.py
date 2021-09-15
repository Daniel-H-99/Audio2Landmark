from torch.optim import lr_scheduler
# import torchvision.utils as vutils
import torch
import os
import numpy as np
import math
import yaml
import torch.nn.init as init
import time
#import librosa
import cv2
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init


def recon_lip(params):
    ref_indice = np.array([3] * 4 + [9] * 3 + [14] * 3 + [18] * 2)
    rel_indice = list(range(4)) + list(range(-2, 1)) + list(range(3)) + list(range(-1, 1))

    lip = [-1] * 22
    for i in range(len(params) - 2):
        param = params[i]
        ref_index = ref_indice[i]
        rel_index = rel_indice[i]
        lip[ref_index - rel_index] = np.array([-param[0], param[1]])
        lip[ref_index + rel_index] = param
    
    lip[-2] = params[-2]
    lip[-1] = params[-1]

    return np.stack(lip, axis=0)
        
def fit_lip_to_face(kp, normed_lip, tilt, mean):
    ref =  extract_ref(kp)[0]
    src = np.linalg.norm(normed_lip[-1] - normed_lip[-2], 2)
    scale = ref / src
    scaled_lip = normed_lip * scale
    
    prev_kp = kp - mean
    x, y = prev_kp[:, 0], prev_kp[:, 1]
    c, s = np.cos(-tilt), np.sin(-tilt)
    x_dash, y_dash = x*c + y*s, -x*s + y*c
    prev_kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
    prev_ref_point = (prev_kp_tilt[45] + prev_kp_tilt[36]) / 2
    delta_x = prev_ref_point[1] - (prev_kp_tilt[51] + prev_kp_tilt[57])[1] / 2

    scaled_lip, src_pts = scaled_lip[:-2], scaled_lip[-2:]
    src_pts[:, 1] += delta_x - src_pts.mean(axis=0)[1]
 

    d_low = scaled_lip[9] - prev_kp_tilt[57]
    for i in range(4, 13):
        if i <= 8:
            ratio = (prev_kp_tilt[i][1] - prev_kp_tilt[4][1]) / (prev_kp_tilt[8][1] - prev_kp_tilt[4][1])

        else:
            ratio = (prev_kp_tilt[12][1] - prev_kp_tilt[i][1]) / (prev_kp_tilt[12][1] - prev_kp_tilt[8][1])

        prev_kp_tilt[i] += ratio * d_low

    kp_dn = np.concatenate([scaled_lip, prev_kp_tilt[4:13], src_pts], axis=0)

    x, y = kp_dn[:, 0], kp_dn[:, 1]
    c, s = np.cos(tilt), np.sin(tilt)
    x_dash, y_dash = x*c + y*s, -x*s + y*c
    kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
    kp_tilt, src_pts_tilt = kp_tilt[:-2], kp_tilt[-2:]
    ref_point = (kp[45] + kp[36]) / 2
    src_point = (src_pts_tilt[0] + src_pts_tilt[1]) / 2



    M = ref_point - src_point
    # lip = kp_tilt + mean 
    lip = kp_tilt + M
    lip = lip.astype('int')


    kp[4:13] = lip[-9:]
    kp[-20:] = lip[:-9]
    # new_kp = np.concatenate([kp[:48], lip], axis=0)
    
    return kp

def extract_ref(kp):
    ref_x = np.linalg.norm(kp[45] - kp[36], ord=2, axis=0)
    ref_y = np.linalg.norm(kp[30] - kp[27], ord=2, axis=0)
    return np.array([ref_x, ref_y])

def extract_scale_coeff(kp):
    ref_x = np.linalg.norm(kp[45] - kp[36], ord=2, axis=0)
    ref_y = np.linalg.norm(kp[30] - kp[27], ord=2, axis=0)
    target_x = np.linalg.norm(kp[54] -kp[48], ord=2, axis=0)
    target_y = np.linalg.norm(kp[57] - kp[51], ord=2, axis=0)
    scale = np.array([target_x / ref_x, target_y / ref_y])
    return scale

def normalize_lip(lip):
    ref_x = np.linalg.norm(lip[6] - lip[0], ord=1, axis=0)
    ref_y = np.linalg.norm(lip[9] - lip[3], ord=1, axis=0)
    return lip / np.array([ref_x, ref_y])[np.newaxis]

def mfcc_from_audio(audio_path, n_mfcc, hop_time, window_time, norm=True, eps=0.000001):
    audio, sr = librosa.load(audio_path, sr=None)
    hop_length = int((sr * hop_time) / 1000)
    n_fft = int((sr * window_time) / 1000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
    # Get mean and std
    if norm:
        mean, std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
        for mfcc_idx in range(mfcc.shape[1]):
            mfcc[:,mfcc_idx] = (mfcc[:,mfcc_idx]-mean)/(std+eps)
    return mfcc


# write image to visualize
# def write_image(image_outputs, display_image_num, image_directory, postfix):
#     image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
#     image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
#     vutils.save_image(image_grid, filename='%s/gen_%s.jpg'%(image_directory, postfix), nrow=1)


def save_image(images, image_directory, postfix):
    path_1 = '%s/fake_%s.jpg'%(image_directory, postfix)
    path_2 = '%s/real_%s.jpg' % (image_directory, postfix)
    cv2.imwrite(path_1,images[0])
    cv2.imwrite(path_2,images[1])

def save_image_int(images, image_directory, postfix):
    path_1 = '%s/%s'%(image_directory, postfix)
    cv2.imwrite(path_1,images[0])


# pad mfcc array
def pad_mfcc(mfcc, append=False, target_shape=(100,28)):
    mfcc_pad = np.zeros(target_shape, dtype=np.float32)
    if append:
        mfcc_pad[:mfcc.shape[0],:] = mfcc
    else:
        mfcc_pad[-mfcc.shape[0]:,:] = mfcc
    return mfcc_pad


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_sub_folder(output_directory, is_test=False):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    if not os.path.exists(log_directory):
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)
    if not is_test:
        return checkpoint_directory
    else:
        image_directory = os.path.join(output_directory, 'images')
        if not os.path.exists(image_directory):
            print("Creating directory: {}".format(image_directory))
            os.makedirs(image_directory)
        landmark_directory = os.path.join(output_directory, 'landmarks')
        if not os.path.exists(landmark_directory):
            print("Creating directory: {}".format(landmark_directory))
            os.makedirs(landmark_directory)
        return checkpoint_directory, image_directory, landmark_directory

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
#         train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)
        train_writer.info('[{}] {}: {}'.format(iterationis + 1, m, getattr(trainer, m)))


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f and not 'best' in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def OneHot(pred_score, exc_map):
    one_hot = torch.zeros((pred_score.shape[0], pred_score.shape[1]))
    for i in range(pred_score.shape[0]):
        peak, index = torch.max(pred_score[i], 0)
        one_hot[i][index] = 1
    for i in range(exc_map.shape[0]):
        # one_map = exc_map[i]
        for j in range(exc_map[i].shape[1]):
            a = 2 * one_hot[i][j] - 1
            exc_map[i][:, j] = exc_map[i][:, j] * a

    return exc_map


def Dict_Unite(pretrained_state_dict, model_state_dict, mode):
    if mode == 'video_pretrain':
        for key in pretrained_state_dict:
            if ((key == 'module.model.fc.weight') | (key == 'module.model.fc.bias')):
                pass
            else:
                model_state_dict[key.replace('module.model.', '')] = pretrained_state_dict[key]
    elif mode == 'audio_pretrain':
        for key in pretrained_state_dict:
            if ((key == 'conv8_objs.weight') | (key == 'conv8_objs.bias')) | (
                    (key == 'conv8_scns.weight') | (key == 'conv8_scns.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
    else:
        print('Forgot to set the pretrained model mode')

    return model_state_dict

def write_log(log, output_directory):
    with open(os.path.join(output_directory, 'log.txt'), 'a') as f:
        f.write(log+'\n')


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def draw_heatmap_from_78_landmark(landmark, width, height):
    heatmap = np.zeros((width, height), dtype=np.uint8)
    #print(landmark)

    # draw lines
    def draw_line(start_idx, end_idx):
        for pts_idx in range(start_idx, end_idx):
            cv2.line(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])),
                     (int(landmark[pts_idx * 2 + 2]), int(landmark[pts_idx * 2 + 3])), thickness=3, color=255)

    draw_line(84-84+19, 90-84+19)     # upper outer
    draw_line(96-84+19, 100-84+19)   # upper inner
    draw_line(100-84+19, 103-84+19)   # lower inner
    draw_line(90-84+19, 95-84+19)    # lower outer
    draw_line(0, 18)    # jaw line
    cv2.line(heatmap, (int(landmark[(96-84+19) * 2]), int(landmark[(96-84+19) * 2 + 1])),
             (int(landmark[(103-84+19) * 2]), int(landmark[(103-84+19) * 2 + 1])), thickness=3, color=255)
    cv2.line(heatmap, (int(landmark[(84-84+19) * 2]), int(landmark[(84-84+19) * 2 + 1])),
             (int(landmark[(95-84+19) * 2]), int(landmark[(95-84+19) * 2 + 1])), thickness=3, color=255)
    heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
    return heatmap

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1):

# 	keypoints = np.int(keypoints)
    keypoints = keypoints.astype(int)
# 	print(keypoints)
	# for i in range(48, 59):
	# 	cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
	# cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
	# cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
	# cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
	# cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
	# for i in range(60, 67):
	# 	cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
    cv2.fillPoly(new_img, [keypoints[48:60]], color=(255, 255, 255))
    cv2.fillPoly(new_img, [keypoints[60:68]], color=(0, 0, 0))
    return new_img

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	kp_dn = N * kp_features_mouth
	x, y = kp_dn[:20], kp_dn[20:]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x*c + y*s, -x*s + y*c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
	kp = kp_tilt + mean
	kp = kp.astype('int')
#     print(kp)
	return kp
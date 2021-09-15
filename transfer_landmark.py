from utils import prepare_sub_folder, recon_lip, write_loss, write_log, get_config, Timer, draw_heatmap_from_78_landmark, save_image, drawLips, getOriginalKeypoints, fit_lip_to_face, normalize_lip
from data import get_data_loader_list, default_pickle_loader
import argparse

import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import shutil
import pickle
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, required=True)
parser.add_argument('--target_dir', type=str, required=True)

opts = parser.parse_args()

src_vid = os.path.basename(opts.source_dir)
tgt_vid = os.path.basename(opts.target_dir)

src_kp_path = os.path.join(opts.source_dir, '../rawKp.pickle')
tgt_kp_path = os.path.join(opts.target_dir, '../rawKp.pickle')
result_dir = os.path.join(opts.target_dir, 'landmarks_from_{}'.format(src_vid))
image_dir = os.path.join(opts.target_dir, 'img_transfered_from_{}'.format(src_vid))

os.makedirs(result_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

src_kp = default_pickle_loader(src_kp_path)[src_vid]
tgt_kp = default_pickle_loader(tgt_kp_path)[tgt_vid]

def fid2key(fid):
    return "{:05d}".format(fid + 1)

for i in tqdm(range(len(tgt_kp))):
    key = fid2key(i)
    normed_lip = src_kp[key][0]
    tmp, N, tilt, mean, _, all_kp = tgt_kp[key]
    normed_lip = recon_lip(normed_lip)
    recon_ldmk = fit_lip_to_face(all_kp, normed_lip, tilt, mean)
    # fake_kp = getOriginalKeypoints(normed_kp, N, theta, mean)

    save_name = os.path.join(result_dir, key + '.txt')
    np.savetxt(save_name, recon_ldmk, fmt='%d', delimiter=',')

    img = cv2.imread(os.path.join(image_dir, '../img', key + '.png'))
    img = drawLips(recon_ldmk, img)
    cv2.imwrite(os.path.join(image_dir, key + '.png'), img)

 
print("finished testing")



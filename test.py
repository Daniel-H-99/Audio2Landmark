from utils import prepare_sub_folder, recon_lip, write_loss, write_log, get_config, Timer, draw_heatmap_from_78_landmark, save_image, drawLips, getOriginalKeypoints, fit_lip_to_face, normalize_lip
from data import get_data_loader_list
import argparse
from torch.autograd import Variable
from trainer import LipTrainer
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch
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
# import tensorboardX
import shutil
import pickle
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--checkpoint_lstm', type=str, default='outputs/LipTrainer/audio2exp_eng_best.pt', help='Path to the checkpoint file')
parser.add_argument("--resume", action="store_true")
parser.add_argument("--bfm", action='store_true')
parser.add_argument("--tag", type=str, default='test', help="tag for experiment")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)

root_dir = config['test_root']
model_name = config['trainer']
logging.basicConfig(filename=os.path.join(opts.output_path + "/logs", model_name + '_' + opts.tag + '.log'), level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
output_directory = os.path.join(opts.output_path + "/outputs", model_name + '_' + opts.tag)
checkpoint_directory, image_directory, landmark_directory = prepare_sub_folder(output_directory, is_test=True)
landmark_directory = config['test_root']
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

f = open(os.path.join(root_dir, config['pca_path']), 'rb')
pca = pickle.load(f)

trainer = LipTrainer(config, bfm=opts.bfm)
trainer.to(config['device'])
state_dict_lstm = torch.load(opts.checkpoint_lstm)
trainer.audio2exp.load_state_dict(state_dict_lstm['audio2exp'])

test_loader = get_data_loader_list(config, split='test', bfm=opts.bfm)
iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0

loss_test = 0
cnt = 0 

if opts.bfm:
    output_bfms = []
    
for id, data in enumerate(tqdm(test_loader)):
    if opts.bfm:
        audio = data.to(config['device']).detach()
    if not opts.bfm:
        audio = data[0].to(config['device']).detach()
        target_kp = data[1].to(config['device']).detach()
        items = len(audio)
        N = data[2][0]   # (-, N, theta, mean, ...)
        theta = data[3][0].numpy()
        mean = data[4][0].numpy()
        all_ldmk = data[5][0].numpy()
        frame_id = data[6][0]
        img = data[7][0].numpy()
        vid_name = data[8][0]
    # Main testing code
    with torch.no_grad():
        if not opts.bfm:
            preds = trainer.forward(audio)
            # print("prediction shape: {}".format(preds.shape))
            # reduced_kp, scale_coeff = preds[:, :-2], preds[:, -2:]
            normed_kp = pca.inverse_transform(preds.detach().cpu().numpy())[0]
            # scale_coeff = scale_coeff[0].detach().cpu().numpy()
            # normed_lip = normalize_lip(N * normed_kp.reshape(2, -1).T)
            normed_lip = recon_lip(normed_kp.reshape(2, -1).T)
            # print(normed_lip.shape)
            recon_ldmk = fit_lip_to_face(all_ldmk, normed_lip, theta, mean)
            # fake_kp = getOriginalKeypoints(normed_kp, N, theta, mean)
            vid_landmark_dir = os.path.join(landmark_directory, vid_name, 'landmarks')
            os.makedirs(vid_landmark_dir, exist_ok=True)
            save_name = os.path.join(vid_landmark_dir, frame_id + '.txt')
            np.savetxt(save_name, recon_ldmk, fmt='%d', delimiter=',')

            img = drawLips(recon_ldmk, img)
            cv2.imwrite(os.path.join(image_directory, frame_id + '.png'), img)

            loss_test += items * trainer.criterion_pca(preds, target_kp).to('cpu')
            cnt += items
        else:
            bfms = trainer.forward(audio).detach().cpu().numpy()
            print(bfms)
            output_bfms.append(bfms[0])

        #print(ldmk)
#         fake_heatmap = draw_heatmap_from_78_landmark(fake_ldmk, 384, 384)
#         real_heatmap = draw_heatmap_from_78_landmark(ldmk, 384, 384)

#         image_output = [fake_heatmap, real_heatmap]
#         save_image(image_output, image_directory, "%06d" % (iterations + 1))

#         iterations += 1
#         if iterations >= len(train_loader):
#             sys.exit('Finish testing')
if not opts.bfm:
    loss_test /= cnt
    print("test loss: {:.5f}".format(loss_test))
    logging.info("test loss: {:.5f}".format(loss_test))
else:
    output_bfms = np.array(output_bfms)[np.newaxis, :, :]
    save_name = os.path.join(landmark_directory, 'bfms.pickle')
    with open(save_name, 'wb') as f:
        pkl.dump(output_bfms, f)
print("finished testing")



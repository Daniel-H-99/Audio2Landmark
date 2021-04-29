from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
# import torchvision.transforms as transforms
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument("--finetune", help="Whether it is finetune or pretrain", action='store_true')
parser.add_argument('--gpu_ids', help='Number of GPUs across which to run in parallel', default='2', type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--checkpoint_prefix', help='prefix for checkpoints, e.g. experiment tag', default='', type=str)

args = parser.parse_args()
gpus = [int(id) for id in args.gpu_ids.split(',')]
args.gpus = gpus
args.ngpu = len(gpus)

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, seed=None, shuffle=True):
        self.all_videos = get_image_list(args.data_root, split)
        self.seed = seed
        self.is_test = split == 'val'
        self.shuffle = shuffle
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:05d}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos) * 50

    def __getitem__(self, idx):
        while 1:
#             print("trying")
            if self.seed is not None:
                idx = self.seed % len(self.all_videos) 
            else:
                idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
#             print("video: {}".format(vidname))
            img_names = list(glob(join(vidname, 'crop', '*.png')))
            if len(img_names) <= 3 * syncnet_T:
                print('{}: not enough images'.format(vidname))
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

#             if random.choice([True, False]):
#                 y = torch.ones(1).float()
#                 chosen = img_name
#             else:
#                 y = torch.zeros(1).float()
#                 chosen = wrong_img_name

            rand = random.choice([True, False])
            if self.shuffle and rand:
                y = torch.zeros(1).float()
                chosen = wrong_img_name
            else:
                y = torch.ones(1).float()
                chosen = img_name
        
#             print('chosen name: {}'.format(chosen))

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                print('{}: fail to load window_fnames'.format(chosen))
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
loss_fn = nn.CrossEntropyLoss()

def cosine_loss(a, v, y):
#     d = nn.functional.cosine_similarity(a, v)
#     print(y)
    a = a / a.norm(p=2, dim=1, keepdim=True)
    v = v / v.norm(p=2, dim=1, keepdim=True)
    cosim = a.matmul(v.transpose(0, 1))
    cosim = cosim[torch.eye(len(cosim)).bool()].unsqueeze(1)
    cosim = torch.atanh(cosim)
    loss = logloss(F.sigmoid(cosim), y)
    return loss

def xent_loss(a, v):
    b, d = a.shape
#     print(v.shape)
    output= torch.cat([a, v], dim=1).view(2 * b , d)
    output = output / output.norm(p=2, dim=1, keepdim=True)
    output = output.matmul(output.transpose(0, 1)) / 0.7
    output = output[(1 - torch.eye(output.shape[0])).bool()].view(output.shape[0], output.shape[0] - 1)
    target = torch.cat([torch.arange(b).unsqueeze(1)] * 2, dim=1).flatten(0) * 2
    loss = loss_fn(output, target.to(output.device))
    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_prefix=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    end_epoch = global_epoch + nepochs
    while global_epoch < end_epoch:
        running_loss = 0.
#         print("train_length: {}".format(len(train_data_loader)))
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
#             print(y)
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)
#             print(x.shape)
#             print(mel.shape)
            a, v = model.project(mel, x)
            y = y.to(device)

#             loss = cosine_loss(a, v, y)
            loss = xent_loss(a, v)
            loss.backward()
            optimizer.step()

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                    
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, checkpoint_prefix, global_epoch)

            

            prog_bar.set_description('[{}] Loss: {}'.format(global_step, running_loss / (step + 1)))

        global_epoch += 1

        
def finetune(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_prefix=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    end_epoch = global_epoch + nepochs
    while global_epoch < end_epoch:
        running_loss = 0.
#         print("train_length: {}".format(len(train_data_loader)))
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
#             print(y)
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)
#             print(x.shape)
#             print(mel.shape)
            a, v = model.project(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
#             loss = xent_loss(a, v)
            loss.backward()
            optimizer.step()
            


            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, checkpoint_prefix, global_epoch)
                
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('[{}] Loss: {}'.format(global_step, running_loss / (step + 1)))

        global_epoch += 1
        
def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 50
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model.project(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
#             print(loss)
#             loss = xent_loss(a, v)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def save_checkpoint(model, optimizer, step, checkpoint_dir, prefix, epoch):

    checkpoint_path = join(
        checkpoint_dir, "{}_checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    test_dataset = Dataset('val')

    if args.finetune:
        train_dataset = Dataset('train')
        train_data_loader = data_utils.DataLoader(
            train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
            num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda:{}".format(args.gpus[0]) if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

if args.finetune:
    finetune(device, model, train_data_loader, test_data_loader, optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=200)
else:
    for e in range(200):
        train_dataset = Dataset('train', seed = global_epoch // 10, shuffle=False)
        train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

        train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_prefix=args.checkpoint_prefix,
              checkpoint_interval=hparams.syncnet_checkpoint_interval,
              nepochs=5)
        print("epoch finished")
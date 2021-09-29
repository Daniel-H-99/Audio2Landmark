from utils import prepare_sub_folder, write_loss, write_log, get_config, Timer
from data import get_data_loader_list
import argparse
from torch.autograd import Variable
from trainer import LipTrainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys

import shutil
import logging




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--bfm", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']


trainer = LipTrainer(config, bfm=opts.bfm)

trainer.to(config['device'])

train_loader = get_data_loader_list(config, split='train', bfm=opts.bfm)
eval_loader = get_data_loader_list(config, split='eval', bfm=opts.bfm)

model_name = config['trainer']
# train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
logging.basicConfig(filename=os.path.join(opts.output_path + "/logs", model_name + '.log'), level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = os.path.join(prepare_sub_folder(output_directory), '..')
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0

min_loss = 10000000

prev = None
while True:
    for id, data in enumerate(train_loader):
        trainer.train()
        trainer.update_learning_rate()
        audio = data[0].to(config['device']).detach()   # B x T x C x H x W
        parameter = data[1].to(config['device']).detach()   # B x T x F
        # print(audio.shape)
    
        # Main training code
        loss_pca = trainer.trainer_update(audio, parameter)
#         torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            log = "Iteration: %08d/%08d, Loss_exc: %f" % (iterations + 1, max_iter, loss_pca)
            print(log)
            write_log(log, output_directory)
#             write_loss(iterations, trainer, train_writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)
            
        if (iterations + 1) % config['eval_iter'] == 0:
            loss_eval = 0
            cnt = 0
            with torch.no_grad():
                for id, data in enumerate(eval_loader):
                    audio = data[0].to(config['device']).detach()
                    parameter = data[1].to(config['device']).detach()

                    # Main training code
                    loss = trainer.check_eval_loss(audio, parameter)
                    items = len(data[0])
                    loss_eval += items * loss
                    cnt += items
                    torch.cuda.empty_cache()
                
            loss_eval /= cnt
            log = "[Eval] Iteration: %08d/%08d, Loss_Eval: %f" % (iterations + 1, max_iter, loss_eval)
            print(log)
            write_log(log, output_directory)
            if min_loss > loss_eval:
                min_loss = loss_eval
                log = "[Eval] Iteration: %08d/%08d, model saved" % (iterations + 1, max_iter)
                print(log)
                write_log(log, output_directory)
                trainer.save(checkpoint_directory, iterations, is_best=True)
                
        iterations += 1
        
        if iterations >= max_iter:
            sys.exit('Finish training')

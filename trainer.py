import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
from utils import OneHot, weights_init, get_model_list, get_scheduler, Dict_Unite
from networks import Audio2Exp
from encoder import Encoder



class LipTrainer(nn.Module):
    def __init__(self, param, bfm=False, is_train = True):
        super(LipTrainer, self).__init__()
        lr = param['lr']
        # Initiate the networks

        self.audio2exp = Audio2Exp(param, is_train=is_train)

        if param['load_from_wav2lip']:
            self.audio2exp.encoder.load_state_dict(torch.load('a2l/audio_encoder.pth'))
#         self.audio2exp = Audio2Exp(param)
        a2e_params = list(self.audio2exp.parameters())

        self.a2e_opt = torch.optim.Adam([p for p in a2e_params if p.requires_grad], lr=lr,
                                        betas=(param['beta1'], param['beta2']), weight_decay=param['weight_decay'])

#         self.a2e_opt = torch.optim.SGD([p for p in a2e_params if p.requires_grad], lr=0.003,
#                                         weight_decay=param['weight_decay'])

        self.a2e_scheduler = get_scheduler(self.a2e_opt, param)

        self.momentum_weight = param['momentum_weight']
        # Network weight initialization
        self.apply(weights_init(param['init']))

        self.device = param['device']
        
    def criterion_pca(self,input, target):
#         return torch.mean(F.pairwise_distance(input.double(), target.double())).to(self.device)
        return nn.MSELoss()(input.double(), target.double())

    def forward(self, audio):
        self.eval()
        exp_pred = self.audio2exp(audio)
        return exp_pred

    def check_eval_loss(self, audio, parameter):
        # change B x T x C x H x W to B x T x C x H x W
        self.eval()

        with torch.no_grad():
            b, t, c, h, w = audio.shape
            audio = audio.view(b*t, h, w).transpose(1, 2)
            parameter = parameter.view(b*t, -1) 

            pca_pred = self.audio2exp(audio)
            self.loss_exc = self.criterion_pca(pca_pred, parameter)
            # pca_pred = pca_pred.view(b, t, -1)
            # parameter = parameter.view(b, t, -1)
            # momentum_pred = (pca_pred[:, 1:] - pca_pred[:, :-1]).view(b * (t - 1), -1)
            # momentum = (parameter[:, 1:] - parameter[:, :-1]).view(b * (t - 1), -1)
            # similarity = F.cosine_similarity(momentum_pred, momentum).mean()
            # self.loss_exc -= self.momentum_weight * similarity
            # self.loss_exc += self.momentum_weight * torch.nn.L1Loss()(momentum_pred, torch.zeros_like(momentum_pred))
        return self.loss_exc

    def trainer_update(self, audio, parameter):
        # change B x T x C x H x W to B x T x C x H x W
        self.train()

        b, t, c, h, w = audio.shape
        audio = audio.view(b*t, h, w).transpose(1, 2)
  
        parameter = parameter.view(b*t, -1) 

        self.a2e_opt.zero_grad()
        pca_pred = self.audio2exp(audio)
        self.loss_exc = self.criterion_pca(pca_pred, parameter)
        # pca_pred = pca_pred.view(b, t, -1)
        # parameter = parameter.view(b, t, -1)
        # momentum_pred = (pca_pred[:, 1:] - pca_pred[:, :-1]).view(b * (t - 1), -1)
        # momentum = (parameter[:, 1:] - parameter[:, :-1]).view(b * (t - 1), -1)
        # # similarity = F.cosine_similarity(momentum_pred, momentum).mean()
        # # self.loss_exc -= self.momentum_weight * similarity
        # # print(momentum_pred.shape)
        # self.loss_exc += self.momentum_weight * torch.nn.L1Loss()(momentum_pred, torch.zeros_like(momentum_pred))
        self.loss_exc.backward()
        self.a2e_opt.step()

        return self.loss_exc

    def update_learning_rate(self):
        if self.a2e_scheduler is not None:
            self.a2e_scheduler.step()


    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, 'audio2exp')
        state_dict = torch.load(last_model_name)
        self.audio2exp.load_state_dict(state_dict['audio2exp'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.a2e_opt.load_state_dict(state_dict['a2e_opt'])

        # Reinitilize schedulers
        self.a2e_scheduler = get_scheduler(self.a2e_opt, param, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations
    
    def save(self, snapshot_dir, iterations, is_best=False):
        # Save generators, discriminators, and optimizers
        if is_best:
            a2e_name = os.path.join(snapshot_dir, 'audio2exp_best.pt')
        else:
            a2e_name = os.path.join(snapshot_dir, 'audio2exp_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'audio2exp': self.audio2exp.state_dict()}, a2e_name)
        torch.save({'a2e_opt': self.a2e_opt.state_dict()}, opt_name)

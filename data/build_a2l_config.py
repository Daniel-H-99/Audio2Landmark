import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets/face/train', help='Path to the train data file.')
opts = parser.parse_args()

with open(os.path.join(opts.data_root, 'config.txt'), 'w') as f:
    videos = os.listdir(opts.data_root)
    for video in videos:
        if not '.mp4' in video:
            continue
        print("processing {}...".format(video))
        kps = sorted(os.listdir(os.path.join(opts.data_root, video, 'keypoints')))
        for kp in kps:
            if kp == '.ipynb_checkpoints':
                continue
            index = kp.replace('.txt', '')
            audio = os.path.join(opts.data_root, video, 'audio', '{:05d}'.format(int(index) - 1) + '.pickle')
            if os.path.exists(audio):
                f.write(video + '/' + kp.replace('.txt', '') + '\n')

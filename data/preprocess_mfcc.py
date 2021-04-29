import os
import sys
import pickle
import librosa
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy, cv2, os, sys, argparse

parser = argparse.ArgumentParser(description='landmark detector')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='datasets/face/desk', type=str)

# parser.add_argument("--", help="Root folder of the preprocessed LRS2 dataset", default='datasets/face/desk', type=str)
# parser.add_argument("--vid_name", default='test_0_0.mp4', type=str)

args = parser.parse_args()
config = {
    'sample_interval': 1 / 100,
    'window_len': 0.02,
    'n_mfcc': 40,
    'fps': 25,
    'input_dir': args.data_root
}

# def crop_audio_window(self, spec, start_frame):
#     # num_frames = (T x hop_size * fps) / sample_rate
#     start_frame_num = self.get_frame_id(start_frame)
#     start_idx = int(80. * (start_frame_num / float(hparams.fps)))

#     end_idx = start_idx + syncnet_mel_step_size

#     return spec[start_idx : end_idx, :]
# get mfcc feature from audio file
def mfcc_from_file(audio_path, n_mfcc=config['n_mfcc'], sample_interval=config['sample_interval'], window_len=config['window_len']):
    # load audio to time serial wave, a 2D array
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=int(sample_interval*sr), n_fft=int(window_len*sr))
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs = np.concatenate((mfccs, mfccs_delta), axis=0)
    # MFCC mean and std
    mean, std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
    mfccs = (mfccs-mean)/std
    padding_front = np.zeros((2 * n_mfcc, 10))
    padding_back = np.zeros((2 * n_mfcc, 9))
    front = np.column_stack((padding_front, mfccs))
    mfccs = np.column_stack((front, padding_back))
    return mfccs

def split_input_target(mfccs):
    #assert mfccs.shape[1] == parameters.shape[1], 'Squence length in mfcc and parameter is different'
    #assert phonemes.shape[1] == parameters.shape[1], 'Squence length in phoneme and parameter is different'
    seq_len = mfccs.shape[1]
    inputs = mfccs
    # target: parameter at a time, input: silding window (past 80, future 20)
    input_list = []
    for idx in range(10, seq_len - 9, 4):
        input_list.append(inputs[:, idx-10:idx+10])
    return input_list

if __name__ == '__main__':
    data_root = config['input_dir']
    videos = os.listdir(data_root)
    for video in videos:
        if not '.mp4' in video:
            continue
        audio_path = os.path.join(data_root, video, 'audio.wav')# , sent['parameter']
        audio_dir = os.path.join(data_root, video, 'audio')
        print("processing {}...".format(audio_path))
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
#         try:
#             wav = audio.load_wav(audio_path, hparams.sample_rate)
#             orig_mel = audio.melspectrogram(wav)
#         except Exception as e:
#             continue

#         padding_front = np.zeros((80, 3))
#         padding_back = np.zeros((80, 0))
#         front = np.column_stack((padding_front, mfccs))
#         mfccs = np.column_stack((front, padding_back))
    
#         mel = self.crop_audio_window(orig_mel.copy(), img_name)
        
#         for
        mfccs = mfcc_from_file(audio_path)
        input_list = split_input_target(mfccs)
#         name_prefix = audio.replace('.wav', '') # maybe also ".mp3" and other files
        for idx in range(len(input_list)):
            input_data = input_list[idx]
            input_path = os.path.join(data_root, video, 'audio', '%05d.pickle'%(idx))

            with open(input_path, 'wb') as f:
                pickle.dump(input_data.T, f)




























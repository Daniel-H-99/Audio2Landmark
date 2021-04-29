import os
import glob
from skimage import io
import numpy as np
import dlib
import sys
import os, random, cv2, argparse

# if len(sys.argv) < 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'test'):
#     raise ValueError('usage: python data/face_landmark_detection.py [train|test]')

parser = argparse.ArgumentParser(description='landmark detector')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='datasets/face/desk', type=str)

parser.add_argument("--vid_name", default='test_0_0.mp4', type=str)

args = parser.parse_args()

# parser.add_argument('--gpu_ids', help='gpu ids to use e.g. 0,1,2', default='2', type=str)

dataset_path = args.data_root
videos = os.listdir(dataset_path)
for video in videos:
    if not '.mp4' in video:
        continue
    faces_folder_path = os.path.join(dataset_path, video, 'img')
    predictor_path = os.path.join(dataset_path, 'shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img_paths = sorted(glob.glob(faces_folder_path + '*'))
    for i in range(len(img_paths)):
        f = img_paths[i]
        print("Processing video: {}".format(f))
        save_path = os.path.join(dataset_path, video, 'keypoints')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for img_name in sorted(glob.glob(os.path.join(f, '*.png'))):
            img = io.imread(img_name)
            dets = detector(img, 1)
            if len(dets) > 0:
                shape = predictor(img, dets[0])
                points = np.empty([68, 2], dtype=int)
                for b in range(68):
                    points[b,0] = shape.part(b).x
                    points[b,1] = shape.part(b).y

                save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
                np.savetxt(save_name, points, fmt='%d', delimiter=',')

import numpy as np
import sys
import os
from sklearn.decomposition import PCA
import pickle as pkl
from tqdm import tqdm
import scipy, cv2, os, sys, argparse

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='landmark detector')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='datasets/face/desk', type=str)

# parser.add_argument("--vid_name", default='test_0_0.mp4', type=str)

args = parser.parse_args()


def getTilt(keypointsMean):
    # Remove in plane rotation using the eyes
    eyes = np.array(keypointsMean[36:48])
    x = eyes[:, 0]
    y = -1 * eyes[:, 1]
    # print('X:', x)
    # print('Y:', y)
    m = np.polyfit(x, y, 1)
    tilt = np.degrees(np.arctan(m[0]))
    return tilt

def getKeypointFeatures(keypoints):
    # Mean Normalize the keypoints wrt the center of the mouth
    # Leads to face position invariancy
    mouth_kp_mean = np.average(keypoints[48:68])
    keypoints_mn = keypoints - mouth_kp_mean

    # Remove tilt
    x_dash = keypoints_mn[:, 0]
    y_dash = keypoints_mn[:, 1]
    theta = np.deg2rad(getTilt(keypoints_mn))
    c = np.cos(theta);	s = np.sin(theta)
    x = x_dash * c - y_dash * s	# x = x'cos(theta)-y'sin(theta)
    y = x_dash * s + y_dash * c # y = x'sin(theta)+y'cos(theta)
    keypoints_tilt = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))

    # Normalize
    N = np.linalg.norm(keypoints_tilt, 2)
    return [keypoints_tilt/N, N, theta, mouth_kp_mean]

d = {}
saveFilename = os.path.join(args.data_root, 'rawKp.pickle')

# fileDir = os.path.dirname(os.path.realpath('__file__'))


# file = args.vid_name
desk = args.data_root
videos = os.listdir(desk)
# print("file name: {}".format(file))



# kp_files = os.listdir(fileDir)
for h in range(len(videos)):
    video = videos[h]
    if not '.mp4' in video:
        continue
    kp_dir = os.path.join(desk, video, 'keypoints')
    kp_files = sorted(os.listdir(kp_dir))
    bigList = {}
    next_index = 1
    print("processing {}...".format(kp_dir))
    for i in range(len(kp_files)):
        kp_file = kp_files[i]
        if kp_file == '.ipynb_checkpoints':
            continue
        index = int(kp_file.replace('.txt', ''))
        if next_index != index:
            print('{} missing'.format(video + '/' + str(next_index)))
            next_index = index
        next_index += 1
        temp = np.array(np.loadtxt(open(os.path.join(kp_dir, kp_file), "rb"), delimiter=",", skiprows=0)).astype("float")

        keypoints = temp.reshape(68, 2)

        #print keypoints

        mouthMean = np.average(keypoints[48:68], 0)
        keypointsMean = keypoints - mouthMean

        xDash = keypointsMean[:, 0]
        yDash = keypointsMean[:, 1]

        theta = np.deg2rad(getTilt(keypointsMean))

        c = np.cos(theta);	
        s = np.sin(theta)

        x = xDash * c - yDash * s	# x = x'cos(theta)-y'sin(theta)
        y = xDash * s + yDash * c   # y = x'sin(theta)+y'cos(theta)

        keypointsTilt = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))

        # Normalize
        N = np.linalg.norm(keypointsTilt, 2)

        #print N
        keypointsNorm = keypointsTilt / N

        kpMouth = keypointsNorm[48:68]
        storeList = [kpMouth, N, theta, mouthMean, keypointsNorm, keypoints]
        bigList['{:05d}'.format(index)] = storeList
#         print("added")
    d[video] = bigList
print(d.keys())
with open(saveFilename, "wb") as outputFile:
	pkl.dump(d, outputFile)

print("normed keypoint dumped")

bigList = []
newList = []


if (os.path.exists(saveFilename)):
	with open(saveFilename, 'rb') as outputFile:
		bigList = pkl.load(outputFile)

mkps = {}
print(bigList.keys())
for key in tqdm(sorted(bigList.keys())):
    mkps[key] = {}
    for k in bigList[key].keys():
        frameKp = bigList[key][k]
        kpMouth = frameKp[0]
        x = kpMouth[:, 0].reshape((1, -1))
        y = kpMouth[:, 1].reshape((1, -1))      ###### weights for y #####
        X = np.hstack((x, y)).reshape((-1))
        newList.append(X.tolist())
        mkps[key][k] = X
    
X = np.array(newList)

pca = PCA(n_components = 20)
pca.fit(X)
with open(os.path.join(args.data_root, 'PCA.pickle'), 'wb') as file:
	pkl.dump(pca, file)
	
with open(os.path.join(args.data_root, 'PCA_explanation.pickle'), 'wb') as file:
	pkl.dump(pca.explained_variance_ratio_, file)

print('Explanation for each dimension:', pca.explained_variance_ratio_)
print('Total variance explained:', 100 * sum(pca.explained_variance_ratio_))

"""
upsampledKp = {}
for key in tqdm(sorted(bigList.keys())):
	print('Key:', key)
	nFrames = len(bigList[key])
	factor = int(np.ceil(100/25))
	# Create the matrix
	newUnitKp = np.zeros((int(factor * nFrames), bigList[key][0][0].shape[0], bigList[key][0][0].shape[1]))
	newKp = np.zeros((int(factor*nFrames), bigList[key][0][-1].shape[0], bigList[key][0][-1].shape[1]))

	print('Shape of newUnitKp:', newUnitKp.shape, 'newKp:', newKp.shape)
	for idx, frame in enumerate(bigList[key]):
		newKp[(idx*(factor)), :, :] = frame[-1]
		newUnitKp[(idx*(factor)), :, :] = frame[0]

		if (idx > 0):
			start = (idx - 1) * factor + 1
			end = idx * factor
			for j in range(start, end):
				newKp[j, :, :] = newKp[start-1, :, :] + ((newKp[end, :, :] - newKp[start-1, :, :]) * (np.float(j+1-start)/np.float(factor)))
				l = getKeypointFeatures(newKp[j, :, :])
				newUnitKp[j, :, :] = l[0][48:68, :]
		
	upsampledKp[key] = newUnitKp
"""

# Use PCA to de-correlate the points

up = {}
reduced = {}
keys = sorted(mkps.keys())
# print(bigList.keys())
for key in tqdm(keys):
#     print(key)
    up[key] = {}
    reduced[key] = {}
    for k in mkps[key].keys():
        mouth_kp = mkps[key][k]
#     print(mouth_kp.shape)
#     print(mouth_kp)

        up[key][k] = mouth_kp
    
        XTrans = pca.transform(mouth_kp.reshape(1, -1))
        reduced[key][k] = XTrans
#     print(XTrans.shape)

with open(os.path.join(args.data_root, 'mouthKp.pickle'), 'wb') as file:
    pkl.dump(up, file)

with open(os.path.join(args.data_root, 'PCA_reducedKp.pickle'), 'wb') as file:
    pkl.dump(reduced, file)

print('Saved Everything')
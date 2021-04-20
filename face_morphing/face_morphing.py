import os,json,argparse
from collections import OrderedDict
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
from utils import read_img, save_img, inside_tri, check_dir
from config import LANDMARK_NAMES, TARGET2_LANDMARKS
from tqdm import tqdm 

import imageio

save_dir = './results1/'
check_dir(save_dir)

def get_landmarks(filename):
	img_name = filename.split('.')[0]
	
	# User: yueyang2000
	if not os.path.exists(img_name+'.json'):
		os.system('curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key=DL0TMmw5MIZyUqm8gvrn0VxpFPXCJf7J" -F "api_secret=46agzt3uQFsyAe0PiixMQN4BgpFxge_m" -F "image_file=@%s" -F "return_landmark=1" > %s.json'%(filename,img_name))
	dic=OrderedDict(json.loads(open(img_name+'.json').read()))

	try:
		landmarks = dic['faces'][0]['landmark']
	except:
		# no face detected, manual landmark
		landmarks = TARGET2_LANDMARKS

	img = Image.open(filename)
	# add four angles
	landmarks['lt'] = {'x': 0, 'y': 0}
	landmarks['rt'] = {'x': img.size[0]-1, 'y': 0}
	landmarks['lb'] = {'x': 0, 'y': img.size[1]-1}
	landmarks['rb'] = {'x': img.size[0]-1, 'y': img.size[1]-1}


	draw = ImageDraw.Draw(img)
	for l in LANDMARK_NAMES:
		# print(l)
		x, y = landmarks[l]['x'], landmarks[l]['y']
		draw.ellipse((x-1,y-1,x+1,y+1), fill=(255, 0, 0), outline=(255, 0, 0))
	img.save(os.path.join(save_dir, img_name + '_landmark.png'))
	
	return np.array([[landmarks[l]['y'], landmarks[l]['x']] for l in LANDMARK_NAMES])



def visualize_delaunay(tri, landmarks, img, name):
	img1 = Image.fromarray(img)
	draw = ImageDraw.Draw(img1)
	for i in range(tri.shape[0]):
		indices = tri[i]
		draw.line((landmarks[indices[0]][1], landmarks[indices[0]][0], landmarks[indices[1]][1], landmarks[indices[1]][0]))
		draw.line((landmarks[indices[0]][1], landmarks[indices[0]][0], landmarks[indices[2]][1], landmarks[indices[2]][0]))
		draw.line((landmarks[indices[1]][1], landmarks[indices[1]][0], landmarks[indices[2]][1], landmarks[indices[2]][0]))
	img1.save(os.path.join(save_dir, name))

def get_affine_matrix(t1, t2):
    # A = TB, T = A*B^-1
    A = np.concatenate((t1.transpose(), np.ones((1,3))), axis=0)
    B = np.concatenate((t2.transpose(), np.ones((1,3))), axis=0)
	# backward morph
    return A.dot(np.linalg.inv(B))

def get_tri_mask(t, shape):
	max_cor = np.max(t, axis=0).astype(np.int)
	min_cor = np.min(t, axis=0).astype(np.int)
	result = np.zeros(shape)
	for x in range(min_cor[0], max_cor[0]+1):
		for y in range(min_cor[1], max_cor[1]+1):
			if inside_tri(t, np.array([x,y])):
				result[x, y] = 1
	return result

def affine_transform(img, t1, t):
	affine = get_affine_matrix(t1, t)
	result = img.copy()
	max_cor = np.max(t, axis=0).astype(np.int)
	min_cor = np.min(t, axis=0).astype(np.int)
	for x in range(min_cor[0], max_cor[0]+1):
		for y in range(min_cor[1], max_cor[1]+1):
			if inside_tri(t, np.array([x,y])):
				back_cor = affine.dot(np.array([[x],[y],[1]]))
				i, j = int(back_cor[0][0]), int(back_cor[1][0])
				if i >= img.shape[0]: i=img.shape[0]-1
				if i < 0: i = 0
				if j >= img.shape[1]: j=img.shape[1]-1
				if j < 0: j = 0
				result[x, y] = img[i, j]
	return result


def morph_tri(img1, img2, img, t1, t2, alpha):
	# average shape of the triangle
	t = t1 * (1-alpha) + t2 * alpha
	mask = get_tri_mask(t, img.shape)
	# A = TB, T = A*B^-1
	result1 = affine_transform(img1, t1, t)
	result2 = affine_transform(img2, t2, t)
	result = result1 * (1-alpha) + result2 * alpha

	img = img * (1-mask) + result * mask
	return img

def generate_morph_sequence(img1, img2, landmarks1, landmarks2, tri, step=5):
	
	seq = []
	for i in range(0, step+1):
		print(f'generating {i}th image')
		alpha = i / step
		img = np.zeros_like(img1)
		for tri_idx in tqdm(range(tri.shape[0])):
			cur_tri = tri[tri_idx]
			t1 = landmarks1[cur_tri]
			t2 = landmarks2[cur_tri]
			img = morph_tri(img1, img2, img, t1, t2, alpha)
		save_img(img, os.path.join(save_dir, 'step' + str(i) + '.png'))
		seq.append(Image.fromarray(img.astype(np.uint8)))
	return seq

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Face Morphing')
	parser.add_argument('-i1', dest='img1', type=str, help='Source image')
	parser.add_argument('-i2', dest='img2', type=str, help='Target image')
	args=parser.parse_args()

	src = read_img(args.img1)
	print(src.shape)
	tgt = read_img(args.img2)
	assert src.shape == tgt.shape, 'Please input images of the same size!'

	landmarks1 = get_landmarks(args.img1)
	landmarks2 = get_landmarks(args.img2)

	tri = Delaunay(landmarks1).simplices
	visualize_delaunay(tri, landmarks1, src,'source_dealaunay.png')
	visualize_delaunay(tri, landmarks2, tgt,'target_dealaunay.png')
	seq = generate_morph_sequence(src, tgt, landmarks1, landmarks2, tri)
	imageio.mimsave(os.path.join(save_dir, 'result.gif'), seq, 'GIF', duration = 0.5)









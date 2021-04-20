import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d


def save_img(img, name):
    Image.fromarray(img).save(name)


def read_img(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img).copy()

def read_mask(path):
    mask = Image.open(path).convert('L')
    mask = np.asarray(mask).copy()
    return mask > 50

def get_laplacian(a):
    M, N = a.shape 
    pad = np.zeros((M+2, N+2))
    pad[1:M+1, 1:N+1] = a
    result = np.zeros_like(a)
    for i in range(M):
        for j in range(N):
            # around pad[i+1, j+1]
            result[i,j] = pad[i+1,j] + pad[i+1,j+2] + pad[i,j+1] + pad[i+2,j+1] - 4 * pad[i+1,j+1]
    return result


def poisson_image_edit(src, target, mask):
    M, N = mask.shape
    src = src.astype(np.float)
    
    target = target.astype(np.float)
    result = target.copy()
    # lap_mask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    num_unknown = mask.sum()

    lut = {}
    cnt = 0
    for i in range(M):
        for j in range(N):
            if mask[i, j]:
                lut[(i,j)] = cnt 
                cnt += 1
                # result[i,j] *= 0
    # return result
    for c in range(3):
        tar_c = target[:,:,c]
        src_c = src[:, :, c]

        A = np.zeros((num_unknown, num_unknown))
        b = np.zeros(num_unknown)
        lap = get_laplacian(src_c)
        cnt = 0
        for i in range(M):
            for j in range(N):
                if mask[i,j]:
                    A[cnt, cnt] = 4

                    if not mask[i-1, j]:
                        b[cnt] = tar_c[i-1,j]
                    else:
                        A[cnt, lut[(i-1,j)]] = -1
                    if not mask[i+1, j]:
                        b[cnt] += tar_c[i+1, j]
                    else:
                        A[cnt, lut[(i+1,j)]] = -1
                    if not mask[i, j-1]:
                        b[cnt] += tar_c[i, j-1]
                    else:
                        A[cnt, lut[(i,j-1)]] = -1
                    if not mask[i, j+1]:
                        b[cnt] += tar_c[i, j+1]
                    else:
                        A[cnt, lut[(i,j+1)]] = -1

                    b[cnt] -= lap[i,j]
                    cnt += 1

        x = np.linalg.solve(A, b)

        cnt = 0
        for i in range(M):
            for j in range(N):
                if mask[i,j]:
                    result[i,j,c] = x[cnt]
                    cnt += 1
    
    return np.clip(result,0, 255).astype(np.uint8)


def image_fusion(src, target, mask, pos):
    # find a small box that contains the mask
    row_idx = np.where(mask.sum(axis=1) > 0)[0]
    row_lo, row_hi = row_idx.min()-1, row_idx.max()+2
    col_idx = np.where(mask.sum(axis=0) > 0)[0]
    col_lo, col_hi = col_idx.min()-1, col_idx.max()+2
    # cut images
    mask_box = mask[row_lo:row_hi, col_lo:col_hi]
    src_box = src[row_lo:row_hi, col_lo:col_hi, :]
    target_box = target[pos[0]:pos[0]+row_hi-row_lo, pos[1]:pos[1]+col_hi-col_lo, :]
    edited = poisson_image_edit(src_box, target_box, mask_box)
    # replace part of target with the original ones 
    target[pos[0]:pos[0]+row_hi-row_lo, pos[1]:pos[1]+col_hi-col_lo, :] = edited
    return target



if __name__ == '__main__':
    img_dir = './resource/'
    mask1 = read_mask(os.path.join(img_dir, 'test1_mask.jpg')) #(360, 266)
    src1 = read_img(os.path.join(img_dir, 'test1_src.jpg')) #(360, 266, 3)
    target1 = read_img(os.path.join(img_dir, 'test1_target.jpg')) #(427, 770, 3)
    pos1 = (100, 70)
    result1 = image_fusion(src1, target1, mask1, pos1)
    Image.fromarray(result1).save('fusion_result1.jpg')

    mask2 = read_mask(os.path.join(img_dir, 'test2_mask.png'))
    src2 = read_img(os.path.join(img_dir, 'test2_src.png'))
    target2 = read_img(os.path.join(img_dir, 'test2_target.png'))
    pos2 = (190, 150)
    result2 = image_fusion(src2, target2, mask2, pos2)
    Image.fromarray(result2).save('fusion_result2.png')
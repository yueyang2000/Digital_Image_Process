from PIL import Image
import numpy as np 
from scipy.fft import dctn, idctn
import argparse

def save_img(img, filename):
    Image.fromarray(img.astype(np.uint8), mode='L').save(filename)

def Dct1d(x, s=None, axis=-1):
    return dctn(x, type=2, s=s, norm='ortho', axes=axis)

def IDct1d(x, s=None, axis=-1):
    return idctn(x, type=2, s=s, norm='ortho', axes=axis)

def Dct2d(x, s=None):
    return dctn(x, type=2, s=s, norm='ortho', axes=(0,1))

def IDct2d(x, s=None):
    return idctn(x, type=2, s=s, norm='ortho', axes=(0,1))

def psnr(img1, img2):
    mse = ((img1 - img2)**2).mean()
    return 10 * np.log10(255*255/mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DCT Experiment')
    parser.add_argument('--factor', '-f', type=int, default=2)
    parser.add_argument('--block_sz', '-b', type=int, default=8)
    args = parser.parse_args()

    img = np.asarray(Image.open('./resource/lena.png').convert('L'), dtype=np.int32)
    img -= 128
    # shift range [-128, 127]
    LEN = 328
    FACTOR = args.factor
    BLOCK_SZ = args.block_sz

    print('FACTOR =', FACTOR)
    print('BLOCK_SZ =', BLOCK_SZ)

    # First row then column
    t1 = Dct1d(img, axis=1)
    t1 = t1[:, :LEN//FACTOR]
    t2 = Dct1d(t1, axis=0)
    t2 = t2[:LEN//FACTOR, :]
    it2 = IDct1d(t2, s=LEN, axis=0)
    result1 = IDct1d(it2, s=LEN, axis=1)
    print('DCT-1D: psnr =', psnr(img, result1))
    save_img(result1 + 128, 'DCT-1D.png')

    result2 = np.zeros_like(img)
    for i in range(LEN // BLOCK_SZ):
        for j in range(LEN // BLOCK_SZ):
            block = img[i*BLOCK_SZ:(i+1)*BLOCK_SZ, j*BLOCK_SZ:(j+1)*BLOCK_SZ]
            t = Dct2d(block)
            t = t[:BLOCK_SZ//FACTOR, :BLOCK_SZ//FACTOR]
            it = IDct2d(t, s=[BLOCK_SZ,BLOCK_SZ])
            result2[i*BLOCK_SZ:(i+1)*BLOCK_SZ, j*BLOCK_SZ:(j+1)*BLOCK_SZ] = it 
    print('DCT-2D: psnr =', psnr(img, result2))
    save_img(result2 + 128, 'DCT-2D.png')

from PIL import Image
import numpy as np 
from scipy.fft import dctn, idctn
import argparse
from matrices import MATRICES

def save_img(img, filename):
    Image.fromarray(img.astype(np.uint8), mode='L').save(filename)

def Dct2d(x, s=None):
    return dctn(x, type=2, s=s, norm='ortho', axes=(0,1))

def IDct2d(x, s=None):
    return idctn(x, type=2, s=s, norm='ortho', axes=(0,1))

def psnr(img1, img2):
    mse = ((img1 - img2)**2).mean()
    return 10 * np.log10(255*255/mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DCT Experiment')
    parser.add_argument('--a', '-a', type=float, default=1)
    parser.add_argument('--matrix', '-m', type=str, default='jpeg', help='nikon/canon/jpeg')
    args = parser.parse_args()

    matrix = MATRICES[args.matrix] * args.a
    print('Matrix Type:', args.matrix)
    print('a =', args.a)

    LEN = 328
    img = np.asarray(Image.open('./resource/lena.png').convert('L'), dtype=np.int32)
    img -= 128
    psnr_list = []
    result = np.zeros_like(img)
    for i in range(LEN // 8):
        for j in range(LEN // 8):
            block = img[i*8:(i+1)*8, j*8:(j+1)*8]
            t = Dct2d(block)
            t = np.round(t / matrix) * matrix
            it = IDct2d(t, s=[8,8])
            result[i*8:(i+1)*8, j*8:(j+1)*8] = it 
            psnr_list.append(psnr(block, it))
    print('mean psnr =', np.mean(psnr_list))
    save_img(result + 128, 'quantization.png')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys


def map_lut(img, lut):
    f = lambda x: lut[x]
    transform = np.vectorize(f)
    return transform(img)

def map_lut_per_channel(img, lut1, lut2, lut3):
    c0 = map_lut(img[:,:,0], lut1)
    c1 = map_lut(img[:,:,1], lut2)
    c2 = map_lut(img[:,:,2], lut3)
    img1 = np.zeros_like(img)
    img1[:,:,0], img1[:,:,1], img1[:,:,2] = c0, c1, c2
    return img1

def plot_lut(lut, filename):
    plt.bar(np.arange(256), lut)
    plt.savefig('fig/' + filename)
    plt.close()


def brightness(scale=10):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if scale >= 0:
            if i + scale <= 255:
                lut[i] = i + scale
            else:
                lut[i] = 255
        else:
            if i + scale >=0:
                lut[i] = i + scale
            else:
                lut[i] = 0
    return lut

def contrast(a = 0.5):
    bins = np.arange(256)
    T = lambda x: a*(x-127) + 127
    T = np.vectorize(T)
    val = T(bins).astype(np.int32)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if val[i] < 0:
            lut[i] = 0
        elif val[i] > 255:
            lut[i] = 255
        else:
            lut[i] = val[i]
    return lut

def gamma(g):
    lut = np.arange(256, dtype=np.float)
    lut = 255 * np.power(lut/255, g)
    return lut.astype(np.uint8)

def get_cdf(img, channel):
    hist, _ = np.histogram(img[:,:,channel], 256, (0,256), density=True)
    cdf = np.cumsum(hist)
    # print(cdf)
    return cdf

def hist_eq(img):
    lut1 = get_cdf(img, 0) * 255
    plot_lut(lut1.astype(np.uint8), 'lut1.png')
    lut2 = get_cdf(img, 1) * 255
    plot_lut(lut2.astype(np.uint8), 'lut2.png')
    lut3 = get_cdf(img, 2) * 255
    plot_lut(lut3.astype(np.uint8), 'lut3.png')
    return map_lut_per_channel(img, lut1.astype(np.uint8), lut2.astype(np.uint8), lut3.astype(np.uint8))

def rematch_lut(cdf1, cdf2):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        t = 0
        while(t <= 255 and cdf1[i] >= cdf2[t]): t+= 1
        lut[i] = t-1
        # if np.abs(t-1-i) < 20:
        #     lut[i] = t-1
        # elif t-1>i:
        #     lut[i] = i + 20
        # else:
        #     lut[i] = i - 20
    return lut

def hist_rematch(img1, img2):
    target_cdf = [get_cdf(img2, 0), get_cdf(img2, 1), get_cdf(img2, 2)]
    img_cdf = [get_cdf(img1, 0), get_cdf(img1, 1), get_cdf(img1, 2)]
    luts = [rematch_lut(x, y) for x, y in zip(img_cdf, target_cdf)]

    plot_lut(target_cdf[0], 'rematch_tgt_cdf.png')
    plot_lut(img_cdf[0], 'rematch_img_cdf.png')
    plot_lut(luts[0], 'rematch_lut.png')
    return map_lut_per_channel(img1, *luts)





if __name__ == '__main__':
    img=mpimg.imread('resource/building.jpg').copy()
    
    LUTs = {
    'bright+': brightness(20),
    'bright-': brightness(-20),
    'contrast+': contrast(2.0),
    'contrast-': contrast(0.5),
    'gamma_compress': gamma(0.8),
    'gamma_expand': gamma(1.2)
    }
    for trans, lut in LUTs.items():
        plot_lut(lut, trans + '_lut.png')
        img1 = map_lut(img, lut)
        plt.imsave('result/'+trans+'.jpg', img1)

    plt.imsave('result/histogram_equalization.jpg', hist_eq(img))

    img2 = mpimg.imread('resouce/sky.jpg').copy()
    plt.imsave('result/hist_rematch.jpg', hist_rematch(img, img2))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

source = mpimg.imread('./resource/source.jpg').copy()
target = mpimg.imread('./resource/target.jpg').copy()
print(source.shape, target.shape)
anchor0 = np.array([316, 169])
anchor1 = np.array([194, 192])
anchor2 = np.array([389, 509])

# target[anchor0[0]-5:anchor0[0]+5, anchor0[1]-5:anchor0[1]+5, :] = np.array([255,0,0])
# target[anchor1[0]-5:anchor1[0]+5, anchor1[1]-5:anchor1[1]+5, :] = np.array([255,0,0])
# target[anchor2[0]-5:anchor2[0]+5, anchor2[1]-5:anchor2[1]+5, :] = np.array([255,0,0])
# mpimg.imsave('anchor.jpg', target)

v1 =  anchor1 - anchor0
v2 = anchor2 - anchor0
v1 = v1.astype(np.float)
v2 = v2.astype(np.float)
for i in range(source.shape[0]):
    for j in range(source.shape[1]):
        pos = anchor0 + v1 * (source.shape[0] - i - 1) / source.shape[0] + v2 * j / source.shape[1]
        pos = pos.astype(np.int32)
        try:
            target[pos[0], pos[1], :] = source[i,j,:]
        except:
            print(pos)

mpimg.imsave('result.jpg', target)

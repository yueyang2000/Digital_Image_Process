from PIL import Image
import numpy as np 
import argparse
from tqdm import tqdm

def read_img(filename, mode='RGB'):
    return np.array(Image.open(filename).convert(mode)).astype(np.int)

def save_img(img, filename):
    Image.fromarray(img.astype(np.uint8)).save(filename)

def get_energy(img):
    M, N, _ = img.shape
    energy = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            left = img[i-1,j] if i != 0 else 0
            right = img[i+1,j] if i != M-1 else 0
            up = img[i,j-1] if j != 0 else 0
            down = img[i,j+1] if j != N-1 else 0
            energy[i,j] = (np.abs(left-right) + np.abs(up-down)).sum()
    return energy

def optimal_seam(img, vertical=True):
    e = get_energy(img)
    M, N = e.shape
    if not vertical:
        e = e.transpose()
    
    # find (i, xi)
    dp = np.zeros_like(e)
    for j in range(N):
        dp[0, j] = e[0, j]
    for i in range(1,M):
        for j in range(N):
            if j == 0: dp[i,j] = min(dp[i-1,j], dp[i-1,j+1]) + e[i,j]
            elif j == N-1: dp[i,j] = min(dp[i-1,j], dp[i-1,j-1]) + e[i,j]
            else: dp[i,j] = min(dp[i-1,j-1], dp[i-1,j], dp[i-1,j+1]) + e[i, j]
    max_idx = np.argmax(dp[M-1])
    seam = [max_idx]

    cur_idx = max_idx
    for i in range(M-2, -1, -1):
        if cur_idx == 0: choices = [cur_idx, cur_idx+1]
        elif cur_idx == N-1: choices = [cur_idx-1, cur_idx]
        else: choices = [cur_idx-1, cur_idx, cur_idx+1]
        min_cost = min(*[dp[i, choice] for choice in choices])
        for choice in choices:
            if dp[i, choice] == min_cost:
                cur_idx = choice 
                seam.append(choice)
                break
    seam.reverse()
    return seam

def visualize_seam(img, seam):
    img = img.copy()
    for i, s in enumerate(seam):
        img[i, s] = [255, 0, 0]
    save_img(img, 'seam.png')

def remove_seam(img, seam, vertical=True):
    M, N = img.shape[0], img.shape[1]
    new_img = np.zeros((M, N-1, 3))
    for i in range(M):
        new_img[i, :seam[i]] = img[i, :seam[i]]
        new_img[i, seam[i]:] = img[i, seam[i]+1:]
    return new_img
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Seam Carving')
    parser.add_argument('-f', dest='file', required=True)
    parser.add_argument('-c', dest='carve',type=int, default=10)
    args = parser.parse_args()

    img1 = Image.open(args.file)
    img1 = img1.resize((img1.size[0]-args.carve, img1.size[1]))
    img1.save('resize.jpg')
    
    img = read_img(args.file)
    print(img.shape)

    for i in tqdm(range(args.carve)):
        seam = optimal_seam(img)
        if i == 0: visualize_seam(img, seam)
        img = remove_seam(img, seam)
    print(img.shape)
    seam = optimal_seam(img)
    # visualize_seam(img, seam)
    save_img(img, 'carved.png')


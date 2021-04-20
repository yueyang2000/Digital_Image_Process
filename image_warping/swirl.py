from skimage.transform import swirl
import matplotlib.image as mpimg

grid = mpimg.imread('./resource/grid.jpg').copy()
print(grid.shape)
swirled = swirl(grid, rotation=0, strength=8, radius=800)
mpimg.imsave('./resource/swirl.jpg', swirled)
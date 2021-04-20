import matplotlib.image as mpimg
import numpy as np

city = mpimg.imread('./resource/warping.png').copy()

city_size = city.shape[0] if city.shape[0] < city.shape[1] else city.shape[1]
city = city[:city_size, :city_size, :] 
print(city.shape) # (289, 289, 4)
city_radius = int((city_size - 1) / 2) # range [-144, +144]

# map square to sphere
sphere_radius = 200
img_size = sphere_radius * 2 + 1
sphere = np.zeros((img_size, img_size, 4))
black = np.array([0, 0, 0, 1.])

for i in range(img_size):
    for j in range(img_size):
        x, y = i - sphere_radius, j - sphere_radius
        rho = np.sqrt(x**2 + y**2)
        # not in the circle
        if  rho > sphere_radius:
            sphere[i,j] = black
            continue 
        elif x == 0 and y == 0:
            sphere[i,j,:] = city[city_radius, city_radius,:]
            continue

        phi = np.arcsin(rho / sphere_radius)
        d = city_radius * phi * 2 / np.pi 
        scale = d / rho
        map_x = np.clip(int(x * scale + city_radius), 0, city_size - 1)
        map_y = np.clip(int(y * scale + city_radius), 0, city_size - 1)
        sphere[i, j, :] = city[map_x, map_y, :]


mpimg.imsave('sphere.png', sphere)


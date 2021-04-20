from PIL import Image
import numpy as np 
import os 
def save_img(img, name):
    Image.fromarray(img.astype(np.uint8)).save(name)


def read_img(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img).copy()

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def inside_tri(t, p):
    x1,y1,x2,y2,x3,y3 = tuple(t.reshape(6))
    xp, yp = p[0], p[1]
    c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
    c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
    c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
    return (c1<=0 and c2<=0 and c3<=0) or (c1>=0 and c2>=0 and c3>=0)



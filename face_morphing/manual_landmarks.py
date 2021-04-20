from config import TARGET2_LANDMARKS, LANDMARK_NAMES
from PIL import Image, ImageDraw

img = Image.open('target2.png')
draw = ImageDraw.Draw(img)

landmarks = TARGET2_LANDMARKS

landmarks['lt'] = {'x': 0, 'y': 0}
landmarks['rt'] = {'x': img.size[0]-1, 'y': 0}
landmarks['lb'] = {'x': 0, 'y': img.size[1]-1}
landmarks['rb'] = {'x': img.size[0]-1, 'y': img.size[1]-1}


draw = ImageDraw.Draw(img)
for l in LANDMARK_NAMES:
    # print(l)
    x, y = landmarks[l]['x'], landmarks[l]['y']
    draw.ellipse((x-1,y-1,x+1,y+1), fill=(255, 0, 0), outline=(255, 0, 0))
img.save('target2_landmark.png')
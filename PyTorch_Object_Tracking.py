#!/usr/bin/env python
# coding: utf-8

# #### Importing the Modules

# In[6]:


from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


# #### Configuration

# In[ ]:


config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to("cpu")
model.eval()
classes = load_classes(class_path)
Tensor = torch.FloatTensor


# In[2]:


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


# #### Algorithm for Detecting Boundaries

# In[3]:


#global list to hold the points
points =  []
refPt = []
drawing = False
def click_draw(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, drawing
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        drawing = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        drawing = False
        # draw a rectangle around the region of interest
        cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# **Run the below algorithm 4 times**

# In[4]:


# load the image, clone it, and setup the mouse callback function`
image = cv2.imread('img_0.jpg')
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_draw)
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        points.append(refPt)
        break
cv2.destroyAllWindows()


# In[9]:


# drawing the lines in the image
for point in points:
    print(point[0])
    cv2.line(image,point[0],point[1],(0,255,0),2)
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()


# #### Detection for teams

# In[10]:


import cv2

def getUniqueColors(img):
    '''
    input: any image
    output: unique color values and its count
    job: find out all the unique color values and its count
        in a given image1
    '''
    a = np.copy(img[:,:,:])
    x = np.split(a.ravel(),img.shape[0]*img.shape[1])
    color,color_count = np.unique(x,return_counts= True,axis = 0)
    return color, color_count

def percentageOfColor(img,black = True):
    '''
    add something here tomorrow
    '''
    color,colorcount = getUniqueColors(img)
    black_count = colorcount[0] # black is the lowest color
    percentage_of_black = black_count/(img.shape[0]*img.shape[1])
    if black:
        return percentage_of_black*100
    else:
        return (1 - percentage_of_black)*100
    
    
def color_check(img,color_name='blue'):
    colors = {'blue':
          {'low':np.array([24, 141, 74]),
           'high':np.array([40, 255, 255])
          },'red':
          {
          'low':np.array([71, 54, 0]),
          'high':np.array([180, 154, 255])
          },'green':
          {
           'low':np.array([25,52,72]),
            'high':np.array([102,255,255])
          },'white':
            {
            'low':np.array([171,151,0]),
            'high':np.array([180,255,255])
            }
         }    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    try:
        high = colors[color_name]['high']
        low = colors[color_name]['low']        
    except Exception as e:
        print(f"{e}: color not expected")
        raise NotImplementedError
        
    mask = cv2.inRange(hsv, low, high)
    result = cv2.bitwise_and(img, img, mask=mask)
    return percentageOfColor(result,False)

def colorselector(img,teamcolors):
    '''
    teamcolors = ['blue','white','red']
    index 0 -> team1
    index 1 -> team2
    index 2 -> refree
    teamcolors[posistion]
    '''
    color = [[100,100,0],[0,0,100],[255,0,0]]
    cls = ''
    bestcolor = [color_check(img,i) for i in teamcolors]
    posistion = bestcolor.index(max(bestcolor))
    if posistion == 2:
        cls = 'refree'
    else:
        cls = f"team{posistion+1}"
    return color[posistion],cls


# In[11]:


dict_heatmaps = {'team1':[],'team2':[]}


# #### Checking if points are inside the rectangle (boundary)

# In[88]:


def vector(p1,p2):
    return {'x':(p2[0]-p1[0]),
            'y':(p2[1]-p1[1])}

def dot(u,v):
    return u['x'] * v['x'] + u['y'] * v['y']

def pointInRectangle(m, r):
    AB = vector(r['A'],r['B']);
    AM = vector(r['A'],m)
    BC = vector(r['B'],r['C'])
    BM = vector(r['B'],m)
    dotABAM = AB['x'] * AM['x'] + AB['y'] * AM['y']
    dotABAB = AB['x'] * AB['x'] + AB['y'] * AB['y']
    dotBCBM = BC['x'] * BM['x'] + BC['y'] * BM['y']
    dotBCBC = BC['x'] * BC['x'] + BC['y'] * BC['y']
    return 0 <= dotABAM and dotABAM <= dotABAB and 0 <= dotBCBM and dotBCBM <= dotBCBC


# In[87]:


key = ('A','B','C','D')
point = {'A' : list(points[0][1]) ,
         'B': list(points[1][1]) ,
         'C' : list(points[2][1]),
         'D': list(points[3][1])}
pointInRectangle([293.0, 129.0], point)


# In[89]:


videopath = '2888-3152.mkv'
get_ipython().run_line_magic('pylab', 'inline')
from IPython.display import clear_output

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


# initialize Sort object and video capture
from sort import *
vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
Success = True

def positiveOnly(n):
    if n < 0:
        return 0
    else:
        return n

counter = 0
while Success:
    ret, frame = vid.read()
    Success =  ret
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            # appending my data here
            bbox = tuple(map(positiveOnly,(int(x1),int(y1),int(box_w),int(box_h))))
            # getting the cropped image
            print(bbox)
            if classes[int(cls_pred)] != "foot ball":
                thisimg = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[3]]
                color,cls = colorselector(thisimg,['blue','red','white'])
                # appending the respective midpoints for each team
                midpoint = [x1/2,y1/2]
                # if midpoint in specific range
                print(midpoint)
                contains = pointInRectangle(midpoint, point)
                if contains:
                    dict_heatmaps[cls].append(midpoint)
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                    cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'img3/img_{counter}.jpg',frame)
    counter += 1
    #imshow(frame)
    #fig=figure(figsize=(12, 8))
    #title("Video Stream")
    #imshow(frame)
    #show()
    clear_output(wait=True)
    
#cap.release()
#out.release()
#cv2.destroyAllWindows()


# In[98]:


import matplotlib.pyplot as plt
import numpy as np
x = dict_heatmaps['team1']
y = dict_heatmaps['team2']
plt.scatter()
plt.show()


# In[91]:


from os.path import join, isfile
image_path = 'img3/'
video_path = 'output3_4.mp4'
fps = 40
frames = []
# loading the files
files = [f for f in os.listdir(image_path) if isfile(join(image_path,f))]
files.sort(key= lambda x: int(x.split("_")[1].split(".")[0]))

for i in range(len(files)):
    filename = image_path + files[i]
    image = cv2.imread(filename)
    # getting information about the image
    height,width,layers = img.shape
    size = (width,height)
    frames.append(image)
out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

for i in range(len(frames)):
    out.write(frames[i])
out.release()


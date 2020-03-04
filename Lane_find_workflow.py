import sys
import os
os.chdir(sys.path[0])
sys.path.insert(1, 'helper_functions')
from helper_functions import threshold as thresh
from helper_functions import transform
from helper_functions import calibration
from helper_functions import lane_line_split as lanes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# step 1 read the image
img = cv2.imread(r'test_images\test2.jpg')

# step 2 undistort the image
mtx, dist=calibration.calibrate()
undist=cv2.undistort(img,mtx,dist,None,mtx)

#cv2.imshow('undist',undist)
#cv2.waitKey(0)

# step 3  get the threshold binaries, color, angle, and sobel
bin_col=thresh.color(undist,80,255)
bin_sobel_x=thresh.sobel(undist,20,80,9)
bin_grad_mag=thresh.grad_mag(undist,20,80,9)
bin_angle=thresh.angle(undist,0.7,1.3,15)
combined=np.zeros_like(bin_col)
combined=cv2.bitwise_or(bin_col,bin_sobel_x)
combined=cv2.bitwise_and(bin_angle,combined)
#combined=cv2.bitwise_or(bin_grad_mag,combined)

#cv2.imshow('color',bin_col)
#cv2.imshow('angle',bin_angle)
#cv2.imshow('sobel',bin_sobel)
#cv2.imshow('combined thresh',combined)
#cv2.waitKey(0)

#step 4 apply perspective transfrom
M=transform.transform(mtx, dist)
img_size=(img.shape[1],img.shape[0])
print(img_size)
warped=cv2.warpPerspective(combined,M,img_size)
#cv2.imshow('warped',warped)
#cv2.waitKey(0)

#step 5 plot the lines
bin_out=lanes.fit_polynomial(warped)
Minv=np.linalg.pinv(M)
unwarped=cv2.warpPerspective(bin_out,Minv,img_size)
cv2.imshow('output',unwarped)
cv2.waitKey(0)

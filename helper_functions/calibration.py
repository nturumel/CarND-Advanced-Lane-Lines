import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys

def calibrate():
    images=glob.glob(r'camera_cal/calibration*.jpg')
    objpoints=[]
    imgpoints=[]

    # create real world coordinates
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    #Now iterate through the images
    shape=(cv2.imread(images[0]).shape)
    shape=shape[1::-1]
    for image in images:
        img=cv2.imread(image)
        #Step 1 convert to grey scale
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Step 2 find corners
        ret,corners=cv2.findChessboardCorners(img,(9,6),None)
        if ret ==True:
            objpoints.append(objp)
            imgpoints.append(corners)
            #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #cv2.imshow('corners',img)
            #cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[0::1], None, None)
    return mtx,dist

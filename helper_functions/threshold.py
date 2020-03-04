import cv2
import numpy as np

def color(img,min,max):
    img_hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    img_s=img_hls[:,:,2]
    mask = cv2.inRange(img_s, min, max)
    return mask

def sobel(img,min,max,k_size):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,k_size)
    sobel_mag=np.absolute(sobel_x)
    scaled_sobel=np.uint8(255*(sobel_mag/np.max(sobel_mag)))
    mask = cv2.inRange(scaled_sobel, min, max)
    return mask

def angle(img,min,max,k_size):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,k_size)
    sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,k_size)
    theta=np.arctan2(np.absolute(sobel_y),np.absolute(sobel_x))
    mask=cv2.inRange(theta,min,max)
    return mask

def grad_mag(img,min,max,k_size):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0)
    sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1)
    sobel_mag=np.sqrt((sobel_x)*(sobel_x)+(sobel_y)*(sobel_y))
    scaled_sobel=np.uint8(255*(sobel_mag/np.max(sobel_mag)))
    mask = cv2.inRange(scaled_sobel, min, max)
    return mask

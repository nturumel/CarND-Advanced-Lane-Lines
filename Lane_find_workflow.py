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
import glob
from moviepy.editor import VideoFileClip
# step 1 calibrate the cam and read the image
mtx, dist=calibration.calibrate()
images = glob.glob(r'test_images\test*.jpg')

def ImageProcessPipe(img):
    # step 2 read the image and unidtort
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
    bin_out,left_r,right_r,left_fit_c,right_fit_c,xm_per_pix,ym_per_pix=lanes.fit_polynomial(warped)
    Minv=np.linalg.pinv(M)
    unwarped=cv2.warpPerspective(bin_out,Minv,img_size)

    #step 6 calculate the center
    xMax=img.shape[1]*xm_per_pix
    yMax=img.shape[0]*ym_per_pix

    Center=xMax/2;
    lineLeft=left_fit_c[0]*yMax**2+left_fit_c[1]*yMax+left_fit_c[2]
    lineRight=right_fit_c[0]*yMax**2+right_fit_c[1]*yMax+right_fit_c[2]
    laneMid=(lineLeft+lineRight)/2
    diff=abs(Center-laneMid)

    #step 7 print the radius and the center
    message_curve="left radius = "+str(left_r)+" ,right radius = "+str(right_r)
    message_diff="Vehicle offcenter by: "+str(diff)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org1 = (50, 50)
    org2 = (50, 90)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    #cv2.putText(unwarped,message_curve, org1, font,fontScale, color, thickness, cv2.LINE_AA)
    #cv2.putText(unwarped,message_diff, org2, font,fontScale, color, thickness, cv2.LINE_AA)
    #cv2.imshow('output',unwarped)
    #cv2.waitKey(0)

    #step 8 add weighted back to the original
    output=cv2.addWeighted(img,0.7,unwarped,0.3,0)
    cv2.putText(output,message_curve, org1, font,fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(output,message_diff, org2, font,fontScale, color, thickness, cv2.LINE_AA)
    #cv2.imshow('output',output)
    #cv2.waitKey(0)
    return output


test_output=r'output_images\videooutput.mp4'
clip1=VideoFileClip("challenge_video.mp4")
out_clip=clip1.fl_image(ImageProcessPipe)
out_clip.write_videofile(test_output,audio=False)

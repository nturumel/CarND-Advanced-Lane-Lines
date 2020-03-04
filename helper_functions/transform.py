import cv2
import numpy as np
from matplotlib import pyplot as plt

def transform(mtx, dist):
    img = cv2.imread(r'test_images\straight_lines1.jpg')
    img=cv2.undistort(img,mtx,dist,None,mtx)
    src_p=np.float32([[270,667],[1018.09,667],[496.815,520.8],[788.844,520.8]])
    dst_p=np.float32([[270,667],[1018,667],[270,520.8],[1018,520.8]])
    M=cv2.getPerspectiveTransform(src_p,dst_p)
    return M

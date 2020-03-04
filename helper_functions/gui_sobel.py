from helper_functions import threshold as thresh
from helper_functions import calibration
import cv2

class Tuner:
    def __init__(self,image,kernel_abs=3,kernel_dir=3,min_dir=0,max_dir=255,min_abs=0,max_abs=255):
        self.image=image
        mtx, dist=calibration.calibrate()
        self.image=cv2.undistort(image,mtx,dist,None,mtx)
        self.kernel_abs=kernel_abs
        self.kernel_dir=kernel_dir
        self.min_dir=min_dir
        self.max_dir=max_dir
        self.min_abs=min_abs
        self.max_abs=max_abs

        def onchangeKernelAbs(pos):
            pos=2*pos+1
            self.kernel_abs=pos
            self._render()

        def onchangeKernelDir(pos):
            pos=2*pos+1
            self.kernel_dir=pos
            self._render()

        def onchangeMinAbs(pos):
            self.min_abs=pos
            self._render()

        def onchangeMaxAbs(pos):
            self.max_abs=pos
            self._render()

        def onchangeMinDir(pos):
            self.min_dir=pos
            self._render()

        def onchangeMaxDir(pos):
            self.max_dir=pos
            self._render()


        cv2.namedWindow('Tuner')

        cv2.createTrackbar('kernel_abs', 'Tuner',self.kernel_abs, 10, onchangeKernelAbs)
        cv2.createTrackbar('kernel_dir', 'Tuner',self.kernel_dir, 10, onchangeKernelDir)
        cv2.createTrackbar('min_dir', 'Tuner',self.min_dir, 255, onchangeMinDir)
        cv2.createTrackbar('max_dir', 'Tuner',self.max_dir, 255, onchangeMaxDir)
        cv2.createTrackbar('min_abs', 'Tuner',self.min_abs, 255, onchangeMinAbs)
        cv2.createTrackbar('max_abs', 'Tuner',self.max_abs, 255, onchangeMaxAbs)


        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

    def Kernel_abs(self):
        return self.kernel_abs


    def Kernel_dir(self):
        return self.kernel_dir

    def Min_abs(self):
        return self.min_abs

    def Max_abs(self):
        return self.max_abs

    def Min_dir(self):
        return self.min_dir

    def Max_dir(self):
        return self.max_dir

    def _render(self):
        #cv2.imshow('image',self.image)
        self._sobel_dir=thresh.sobel(self.image,self.min_dir,self.max_dir,self.kernel_dir)
        self._sobel_mag=thresh.grad_mag(self.image,self.min_abs,self.max_abs,self.kernel_abs)
        cv2.imshow('sobel dir',self._sobel_dir)
        cv2.imshow('sobel mag',self._sobel_mag)

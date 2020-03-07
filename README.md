**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Read the image and apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distorted image.png 
[image2]: ./output_images/BE1.jpg 
[image3]: ./output_images/1.jpg 
[video1]: ./project_video.mp4 

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the disortion correction through cv2.undistort(img,mtx,dist,None,mtx) using the values of the mtx and dis obtained in the previous step.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`). I tried a variety of different combinations with the test images and also ended up creating a gui tuner for helping me get the right threshold values for sobel in images. 
In the end I used sobel x the s channel and the direction of sobel.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Since the camera does no change, I picked and image from the test images, and then picked four points in trapezoidal shape, put them in src[], and then estimates how they would look as a sqaure, then used cv2.getPerspectiveTransform(src_p,dst_p) to get the transformation matrix.

```python
    src_p=np.float32([[270,667],[1018.09,667],[496.815,520.8],[788.844,520.8]])
    dst_p=np.float32([[270,667],[1018,667],[270,520.8],[1018,520.8]])
    M=cv2.getPerspectiveTransform(src_p,dst_p)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the slidng window technique to copute the lane line pixels.
![alt text][image2]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the following way:


```python
#Radius:

def fit_polynomial(binary_warped):
    #find the pixels
    leftx,rightx,lefty,righty,out_img=sliding_window(binary_warped)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    #fit the second fit_polynomial
    left_fit=np.polyfit(lefty,leftx,2)
    right_fit=np.polyfit(righty,rightx,2)
    left_poly=np.poly1d(left_fit)
    right_poly=np.poly1d(right_fit)

    #get the y coord
    yplot=np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])

    #plot just for visualisation
    xleft_plot=left_poly(yplot)
    xright_plot=right_poly(yplot)
    
    # calculate the slope
    #convert to real image linspace
    ym_per_pix=30/720
    xm_per_pix=3.7/700

    left_fit_c=np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit_c=np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)

    left_r=((1+2*left_fit_c[0]+left_fit_c[1]**2)**1.5)/np.absolute(2*left_fit_c[0])
    right_r=((1+2*right_fit_c[0]+right_fit_c[1]**2)**1.5)/np.absolute(2*right_fit_c[0])

    #draw the lane lines using fillpoly on out image
    pts_left=np.array([np.transpose(np.vstack([xleft_plot,yplot]))])
    pts_right=np.array([np.flipud(np.transpose(np.vstack([xright_plot,yplot])))])
    pts=np.hstack((pts_left,pts_right))
    cv2.fillPoly(out_img,np.int_([pts]),(0,255,0))

    return out_img,left_r,right_r,left_fit_c,right_fit_c,xm_per_pix,ym_per_pix
```
```python
#Center
    xMax=img.shape[1]*xm_per_pix
    yMax=img.shape[0]*ym_per_pix

    Center=xMax/2;
    lineLeft=left_fit_c[0]*yMax**2+left_fit_c[1]*yMax+left_fit_c[2]
    lineRight=right_fit_c[0]*yMax**2+right_fit_c[1]*yMax+right_fit_c[2]
    laneMid=(lineLeft+lineRight)/2
    diff=abs(Center-laneMid)

```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is the final image

![alt text][image3]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My threshold needs improvemet, right angle intersections, identification of other cars, slow speed

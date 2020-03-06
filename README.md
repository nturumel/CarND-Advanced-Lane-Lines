The Project
The following steps were followed for this project:
1. Calibrate the camera:
  I calibrated the camera with the help of the chess board images present
2. Undistort the image
  I undistorted the image using the matrix values obtained in step 1
3. threshold
  I set the thresholds using  the threshold of color,  soble angle, and absolute sobel
4. apply perrspective transorm
  From the test image I selected a test images, converted a trapezium into a sqaue and got the matrix
5. Plotting Lane lines:
  I found the lane lines on the birds eye view of the road, using the sliding window technique, calculated the lane curvature
6. Calculate the center
  find middle of image, find middle of road, subtract
7. Print
  Display on image
8. add binary to image
  Using addweighted of cv2 I added the outpur image to the original image, we have color

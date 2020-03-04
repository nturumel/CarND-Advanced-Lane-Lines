import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


def sliding_window(binary_warped):
    # take the histogram of the bottom image
    histogram=np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)

    # split the results into right and left
    midpoint=np.int(histogram.shape[0]//2)
    left_x_base=np.argmax(histogram[:midpoint])
    right_x_base=np.argmax(histogram[midpoint:])+midpoint

    #make the windows
    num_win=10
    min_pix=0
    margin=100
    win_height=np.int(binary_warped.shape[0]//num_win)

    #get the non zeros
    nonzeros=binary_warped.nonzero()
    nonzeroy=np.array(nonzeros[0])
    nonzerox=np.array(nonzeros[1])

    #set the starting positions for the windows
    left_x_curr=left_x_base
    right_x_curr=right_x_base

    #create a list to accept indices
    left_indice=[]
    right_indice=[]

    # step through all windows
    for win in range(num_win):
        #make windows
        win_y_high=binary_warped.shape[0]-(win)*win_height
        win_y_low=binary_warped.shape[0]-(win+1)*win_height
        win_xleft_low=left_x_curr-margin
        win_xleft_high=left_x_curr+margin
        win_xright_low=right_x_curr-margin
        win_xright_high=right_x_curr+margin

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        #draw the visualisation
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)

        #check what points lie inside it
        good_left=((nonzeroy<win_y_high)&(nonzeroy>=win_y_low)&(nonzerox<win_xleft_high)&(nonzerox>=win_xleft_low)).nonzero()[0]
        good_right=((nonzeroy<win_y_high)&(nonzeroy>=win_y_low)&(nonzerox<win_xright_high)&(nonzerox>=win_xright_low)).nonzero()[0]

        #append to list
        left_indice.append(good_left)
        right_indice.append(good_right)

        #recenter
        if(len(good_left)>min_pix):
            left_x_curr=np.int(np.mean(nonzerox[good_left]))
        if(len(good_right)>min_pix):
            right_x_curr=np.int(np.mean(nonzerox[good_right]))

    # make a single list

    try:
        left_indice=np.concatenate(left_indice)
        right_indice=np.concatenate(right_indice)
    except ValueError:
        pass

    # Get pixels
    leftx=nonzerox[left_indice]
    lefty=nonzeroy[left_indice]
    rightx=nonzerox[right_indice]
    righty=nonzeroy[right_indice]

    return leftx,rightx,lefty,righty,out_img

def fit_polynomial(binary_warped):
    #find the pixels
    leftx,rightx,lefty,righty,out_img=sliding_window(binary_warped)

    #fit the second fit_polynomial
    left_fit=np.polyfit(lefty,leftx,2)
    right_fit=np.polyfit(righty,rightx,2)
    left_poly=np.poly1d(left_fit)
    right_poly=np.poly1d(right_fit)

    #get the y coord
    yplot=np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])

    #plot
    xleft_plot=left_poly(yplot)
    xright_plot=right_poly(yplot)

    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,255,0]

    plt.plot(xleft_plot,yplot,color='yellow',linewidth=5)
    plt.plot(xright_plot,yplot,color='yellow',linewidth=5)

    plt.imshow(out_img)
    plt.show()
    return out_img

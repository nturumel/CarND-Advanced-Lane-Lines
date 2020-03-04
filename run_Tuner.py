import argparse
import cv2
import os
import sys
from helper_functions import gui_sobel

os.chdir(sys.path[0])
def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()

    img = cv2.imread(args.filename)

    cv2.imshow('input', img)

    tuner_finder = gui_sobel.Tuner(img)

    print ("Tuned parameters:")
    print ("Kernel Absolute Size: %f" % tuner_finder.Kernel_abs())
    print ("Kernel Direction Size: %f" % tuner_finder.Kernel_dir())
    print ("Kernel Absolute Min: %f" % tuner_finder.Min_abs())
    print ("Kernel Absolute Max: %f" % tuner_finder.Max_abs())
    print ("Kernel Direction Min: %f" % tuner_finder.Min_dir())
    print ("Kernel Direction Max: %f" % tuner_finder.Max_dir())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

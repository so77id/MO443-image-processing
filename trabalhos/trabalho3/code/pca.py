
import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

DEBUG=False

GRAPHIC_NAME="graphic"

def create_graphic(graphic_out, graphic0, graphic1, graphic2, llabel0, llabel1, llabel2, ylabel, xlabel):
    ranges = np.arange(graphic0.size)

    plt.plot(ranges, graphic0, 'b-', label=llabel0)
    plt.plot(ranges, graphic1, 'r-', label=llabel1)
    plt.plot(ranges, graphic2, 'g-', label=llabel2)

    plt.legend(loc='upper left', shadow=False)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)

    plt.savefig(graphic_out)
    if(DEBUG):
        plt.show()


def get_filename(path, name, data, extension):
    filename = '{0}/{1}'.format(path, name)
    for d in data:
        filename = '{0}_{1}'.format(filename, d)
    filename = '{0}.{1}'.format(filename, extension)

    return filename


def NMSE (img0, img1):
    return ((img0 - img1) ** 2).mean() / (img0 ** 2).mean()

def RMSE(img0, img1):
    return sqrt(mean_squared_error(img0, img1))

def pca_error(img0, img1):
    error = 0
    _, _, channels = img0.shape

    for i in range(channels):
        error += NMSE(img0[:,:,i],img1[:,:,i])
    return error/channels

def get_pca_image(img_in, k):
    # Get dimensions of image
    height, width, channels = img_in.shape

    # Create new image
    img_out = np.empty((height, width, channels))

    original_size = 0
    new_size = 0
    # for each channel
    for i in range(channels):
        # Get SVD from each channel
        U, s, V = np.linalg.svd(np.float64(img_in[:,:,i]))

        new_U = U[:,0:k]
        new_s = s[0:k]
        new_V = V[0:k,:]

        original_size += U.nbytes + s.nbytes + V.nbytes
        new_size += new_U.nbytes + new_s.nbytes + new_V.nbytes
        # Create new image from each channel
        img_out[:,:,i] = np.dot(new_U, np.dot(np.diag(new_s), new_V))

    # Normalize pixels
    img_out[img_out > 255] = 255
    img_out[img_out < 0] = 0

    # Change image type to uint8
    img_out = img_out.astype('uint8')

    return img_out, (new_size / original_size)

def get_all_pca(img_in, filename_in, path_images_out, path_grapich_out):
    height, _, _ = img_in.shape
    errors = np.array([])
    file_sizes = np.array([])
    memory_sizes = np.array([])
    memory_intersection = None
    file_intersection = None

    original_size = os.stat(filename_in).st_size
    if(DEBUG):
        print("original_size: ", original_size)

    graphic_out = get_filename(path_grapich_out, filename_in.split('/')[-1].split('.')[0], [GRAPHIC_NAME], 'png')

    print("graphic_out: ", graphic_out)


    for i in range(height):
        # Get file name
        filename_out = get_filename(path_images_out, filename_in.split('/')[-1].split('.')[0], [i], filename_in.split('/')[-1].split('.')[-1])
        #Get new image
        img_out, memory_size = get_pca_image(img_in, i)

        # Save memory size of new image
        memory_sizes = np.append(memory_sizes, memory_size)

        # Get error of new image
        error = pca_error(img_in, img_out)
        errors = np.append(errors, error)

        # Save image
        cv2.imwrite(filename_out,img_out)

        # Get size of new image
        file_size = os.stat(filename_out).st_size
        file_size = file_size/original_size
        file_sizes = np.append(file_sizes, file_size)

        if(error <= memory_size and memory_intersection == None):
            memory_intersection = i
            print("Memory memory_intersection  k:", i, "error: ", error)
        if(error <= file_size and file_intersection == None):
            file_intersection = i
            print("Memory file_intersection  k:", i, "error: ", error)

        if(DEBUG):
            print("filename_out: ", filename_out)
            print("k:", i, "Percentage memory:", memory_size)
            print("k:", i,"error: ", error)
            print("k:", i, "Percentage file:", file_size)
            print("------------------")

    create_graphic(graphic_out, errors, file_sizes, memory_sizes,"PCA error", "File size", "Memory size", "%", "K")



def main(argv):
    if (len(argv) < 4):
        print("Incorrect number of arguments")
        print("Execute: python3 pca.py filename_in path_images_out path_grapich_out")
        return -1

    filename_in  = argv[1]
    path_images_out = argv[2]
    path_grapich_out = argv[3]

    img_in = cv2.imread(filename_in, True)

    get_all_pca(img_in, filename_in, path_images_out, path_grapich_out)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
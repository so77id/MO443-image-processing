import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

def bit_spliting(image):
    images = []

    for i in range(8):
        n = 2**i
        new_image = np.bitwise_and(image,n)
        new_image[new_image == n] = 255
        images.append(new_image)

    return images

def normalized_histogram(image, nbins):
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    histr = []
    for i in range(channels):
        hist = cv2.calcHist([image],[i],None,[nbins],[0,256]).flatten()
        hist = hist/hist.sum()
        histr.append(hist)

    return np.array(histr)

def entropy(hist):
    entropy = 0
    width = hist.shape
    for x in range(0, width[0]):\
        entropy = entropy + hist[x] * np.log2(hist[x])

    return -1 * entropy


def main(argv):
    if (len(argv) < 3):
        print("Incorrect number of arguments")
        print("Execute: python3 comparation.py filename_1.png nbins")
        return -1

    filename1 = argv[1]
    nbins = int(argv[2])

    img = cv2.imread(filename1, False)

    images = bit_spliting(img)

    fig = plt.figure("Bit Split")
    i = 1
    for image in images:
        histogram = normalized_histogram(image, nbins).flatten()
        histogram[histogram==0] = 1
        entropy_value = entropy(histogram)
        print("Entropy of "+str(i)+" bit: "+ "{:.5f}".format(entropy_value) )

        a=fig.add_subplot(2,4,i)
        imgplot = plt.imshow(image, cmap='binary')
        a.set_title("Bit "+str(i)+" E: "+"{:.5f}".format(entropy_value) )

        i = i + 1


    fig = plt.figure("Increacing sum")
    i = 1
    string = ""
    image_ = np.zeros(images[0].shape)
    for image in images:
        image[image == 255] = 1
        image_ = image_ + image * 2**i

        a=fig.add_subplot(2,4,i)
        string = string + " " + str(i)
        a.set_title(string)

        imgplot = plt.imshow(image_, cmap='gray')

        i = i + 1


    fig = plt.figure("Decreacing sum")
    j = 7
    i = 1
    string = ""
    image_ = np.zeros(images[0].shape)
    for image in reversed(images):
        image[image == 255] = 1
        image_ = image_ + image * 2**j

        a=fig.add_subplot(2,4,i)
        string = string + " " + str(j+1)
        a.set_title(string)

        imgplot = plt.imshow(image_, cmap='gray')

        i = i + 1
        j = j - 1

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
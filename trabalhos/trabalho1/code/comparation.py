
import sys

import numpy as np
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

display = False

def plot_histogram(hist, nbins, hist_filename):
    bins = np.arange(nbins)

    #TO DO: transform to one plot with three channels
    bgr = ['B', 'G', 'R']
    bgr_names = ['Blue', 'Green', 'Red']
    fig = plt.figure()

    for i in range(hist.shape[0]):
        a=fig.add_subplot(5,1,2*i+1)

        a.bar(bins, hist[i], align='center',
            color=bgr[i], ecolor='black')
        a.set_title("Channel "+bgr_names[i])

    plt.savefig(hist_filename)
    if display:
        plt.show()


def normalized_histogram(image, nbins):
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    histr = []
    for i in range(channels):
        hist = cv2.calcHist([image],[i],None,[nbins],[0,256]).flatten()
        hist = hist/(height * width)
        histr.append(hist)

    return np.array(histr)

def euclidean_distance(hist1, hist2):
    distance = 0.0

    if(hist1.shape == hist2.shape):
        for i in range(hist1.shape[0]):
            distance = distance + np.linalg.norm(hist1[i] - hist2[i], 2)
        distance = distance / hist1.shape[0]

    return distance


def get_distance_between_images(image1, image2, nbins, out_hist1_filename, out_hist2_filename):
    hist1 = normalized_histogram(image1, nbins)
    hist2 = normalized_histogram(image2, nbins)

    distance = euclidean_distance(hist1, hist2)

    plot_histogram(hist1, nbins, out_hist1_filename)
    plot_histogram(hist2, nbins, out_hist2_filename)

    return distance


def main(argv):
    if (len(argv) < 6):
        print("Incorrect number of arguments")
        print("Execute: python3 comparation.py filename_1.png filename_2.png nbins out_hist1_filename out_hist2_filename")
        return -1

    filename1          = argv[1]
    filename2          = argv[2]
    nbins              = int(argv[3])
    out_hist1_filename = argv[4]
    out_hist2_filename = argv[5]

    image1 = cv2.imread(filename1, True)
    image2 = cv2.imread(filename2, True)

    euclidean_distance_value = get_distance_between_images(image1, image2, nbins, out_hist1_filename, out_hist2_filename)

    print("Distance between images: "+str(euclidean_distance_value))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
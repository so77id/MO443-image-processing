
import sys

from scipy import misc
import numpy as np


def merge(img1, img2, percentage1, percentage2):
    return percentage1*img1 + percentage2*img2

def main(argv):
    if (len(argv) < 6):
        print("Incorrect number of arguments")
        print("Execute: python3 file1 file2 percentage1 percentage2 filename_out")
        return -1

    filename1 = argv[1]
    filename2 = argv[2]
    percentage1 = float(argv[3])
    percentage2 = float(argv[4])
    filename_out = argv[5]

    if percentage1 > 1.0 or percentage2 > 1.0 or percentage1 < 0.0 or percentage2 < 0.0:
        print("Percentage must be greater than 0.0 and less than 1.0")
        return -1

    if percentage1 + percentage2 != 1.0:
        print("Percentage1 plus percentaje2 must be 1.0")
        return -1

    image1 = misc.imread(filename1, True)
    image2 = misc.imread(filename2, True)

    if image1.shape != image2.shape:
        print("Images must be the same size")
        return -1

    merged_image = merge(image1, image2, percentage1, percentage2)

    misc.imsave(filename_out, merged_image)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
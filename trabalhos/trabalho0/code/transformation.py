import sys

from scipy import misc
import numpy as np

def negative(image):
    return 255 - image


def transformation(image, initial_range, final_range):
    return np.uint8(((final_range - initial_range)*image)//255 + initial_range)

def main(argv):
    if (len(argv) < 6):
        print("Incorrect number of arguments")
        print("Execute: python3 filename.py filename_in filename_negative filename_transformation initial_range final_range")
        return -1

    filename_in = argv[1]
    filename_negative = argv[2]
    filename_transformation = argv[3]
    initial_range = int(argv[4])
    final_range = int(argv[5])

    image = misc.imread(filename_in, True)

    img_negative = negative(image)
    img_transformation = transformation(image, initial_range, final_range)

    misc.imsave(filename_negative, img_negative)
    misc.imsave(filename_transformation, img_transformation)
    #img_negative.save(filename_negative)
    #img_transformation.save(filename_transformation)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
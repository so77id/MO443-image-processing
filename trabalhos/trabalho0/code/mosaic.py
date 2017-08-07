
import sys
import random

from scipy import misc
import numpy as np

def tiled(img, n_tiles):
    tiles = []
    height, width = img.shape

    for i in range(0,n_tiles):
        for j in range(0,n_tiles):
            row_range_lower = (i * height)//n_tiles
            row_range_upper = (((i + 1) * height)//n_tiles) - 1

            col_range_lower = (j * width)//n_tiles
            col_range_upper = (((j + 1) * width)//n_tiles) - 1

            tiles.append(img[row_range_lower:row_range_upper,col_range_lower:col_range_upper])

    return tiles

def preprocessing(img, n_tiles):
    height, width = img.shape

    new_width = (width // n_tiles) * n_tiles
    new_height = (height // n_tiles) * n_tiles

    new_img = img[0:new_height, 0:new_width]

    return new_img


def create_mosaic(img, n_tiles=4):

    img = preprocessing(img, n_tiles)
    tiles = tiled(img, n_tiles)

    mosaic_order = np.arange(n_tiles**2)
    random.shuffle(mosaic_order)
    mosaic_order = mosaic_order.reshape((n_tiles, n_tiles)) if n_tiles != 4 else np.array([5, 10, 12, 2, 7, 15, 0, 8, 11, 13, 1, 6, 3, 14, 9, 4]).reshape((n_tiles, n_tiles))


    for i in range(0,n_tiles):
        for j in range(0, n_tiles):
            if j == 0:
                new_row = tiles[mosaic_order[i][j]]
            else:
                new_row = np.concatenate((new_row,tiles[mosaic_order[i][j]]), axis=1)
        if i == 0:
            new_img = new_row
        else:
            new_img = np.concatenate((new_img, new_row), axis=0)

    return new_img


def main(argv):
    if (len(argv) < 4):
        print("Incorrect number of arguments")
        print("Execute: python3 filename.py filename_in filename_out number_of_tiles")
        return -1

    filename_in = argv[1]
    filename_out = argv[2]
    n_tiles = int(argv[3])


    image = misc.imread(filename_in, True)
    new_img = create_mosaic(image, n_tiles)

    misc.imsave(filename_out, new_img)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

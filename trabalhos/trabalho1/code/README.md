# Trabalho 1

Requeriments:
* python3
* pip3
* Numpy
* OpenCV3
* matplotlib

The codes of this work are divided in two:

* bit_slicing.py
* comparation.py

To install requeriments run:

`make init`

you need `pip3` installed

## Comparation

This code receives two images to be compared by means of its normalized histograms using Euclidean distance.

The distance is displayed by the console.

To execute:

```
python3 comparation.py path_image1 path_image2 n_bins path_histogram1 path_histogram2
```

Example

```
→ python3 comparation.py ../relatorio/images/photo500x500.png ../relatorio/images/photo.jpg 256 output1.png output2.png

Distance between images: 0.0358778787436
```

## Bit_slicing

This code receives an image and the number of bins to calculate the entropy.

This code receives an image and the number of bins to calculate the entropy. He shows three windows: The first is the eight binary planes, the second is the reconstruction and the image by means of the increasing sum of the planes and the last one is the reconstruction by means of the decreasing sum of the planes.

The entropy of each image is displayed in the image with planes and by console

To execute:

```
python3 bit_slicing.py path_image n_bins
```

```
→ python3 bit_slicing.py ../relatorio/images/rage_me500x500.png 256

Entropy of 1 bit: 1.00000
Entropy of 2 bit: 1.00000
Entropy of 3 bit: 0.99999
Entropy of 4 bit: 0.99999
Entropy of 5 bit: 0.99870
Entropy of 6 bit: 0.97966
Entropy of 7 bit: 0.99497
Entropy of 8 bit: 0.50751
```

# Trabalho 3

Requeriments:
* python3
* pip3
* Numpy
* OpenCV3
* matplotlib

To install requeriments run:

`make init`

you need `pip3` installed

## pca.py

This code receives:
* filename_in
* path_images_out
* path_grapich_out

The grapic image out is displayed into console, also display the intersections. You can change value of DEBUG constant to `True` for display more data about process.

To execute:

```
python3 pca.py filename_in path_images_out path_grapich_out
```

Example

```
→ python3 pca.py ../images/peppers.png  ../outputs ../graphics

graphic_out:  ../graphics/peppers_graphic.png
Memory file_intersection  k: 11 error:  0.716547387161
Memory memory_intersection  k: 103 error:  0.198666492932
```


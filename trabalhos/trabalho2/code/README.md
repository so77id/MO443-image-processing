# Trabalho 2

Requeriments:
* python3
* pip3
* Numpy
* OpenCV3
* matplotlib

The codes of this work are divided in four:

* shot_cut_pixel_difference.py
* shot_cut_blocks.py
* shot_cut_histogram.py
* shot_cut_edges.py

To install requeriments run:

`make init`

you need `pip3` installed

## shot_cut_pixel_difference

This code receives the paths for save the graphic, video and image of summarization, also the thresholds.

The count of shot detected are be displayed into console

To execute:

```
python3 shot_cut_pixel_difference.py video_in path_video_out path_graphic_out path_mosaic_out pixel_threshold frame_threshold
```

Example

```
→  python shot_cut_pixel_difference.py ../videos/lisa.mpg ../video_out ../graph_out ./summarize_out 16 40

video name: ../video_out/lisa_pixel_difference_16_40.avi
graphic name: ../graph_out/lisa_pixel_difference_16_40.png
mosaic name: .//lisa_pixel_difference_16_40.png
Shot detected:  10
```


## shot_cut_blocks

This code receives the paths for save the graphic, video and image of summarization, also the thresholds and the size of blocks.

The count of shot detected are be displayed into console

To execute:

```
python3 shot_cut_blocks.py video_in path_video_out path_graphic_out path_mosaic_out size_of_division error_threshold percentage_of_different_blocks
```

Example

```
→ python3 shot_cut_blocks.py ../videos/lisa.mpg ../video_out/block_difference ../graph_out/block_difference ../resume_out/block_difference 16 90 80

video name: ../video_out/block_difference/lisa_block_difference_16_90_80.avi
graphic name: ../graph_out/block_difference/lisa_block_difference_16_90_80.png
mosaic name: ../resume_out/block_difference/lisa_block_difference_16_90_80.png
Shot detected:  1

```


## shot_cut_histogram

This code receives the paths for save the graphic, video and image of summarization, also the thresholds and the size of bins.

The count of shot detected are be displayed into console

To execute:

```
python3 shot_cut_blocks.py video_in path_video_out path_graphic_out path_mosaic_out nbins alpha_value
```

Example

```
→ python3 shot_cut_histogram.py ../videos/auto.mpg ../video_out/histogram_difference ../graph_out/histogram_difference ../resume_out/histogram_difference 8 13

video name: ../video_out/histogram_difference/auto_histogram_difference_8_13.0.avi
graphic name: ../graph_out/histogram_difference/auto_histogram_difference_8_13.0.png
mosaic name: ../resume_out/histogram_difference/auto_histogram_difference_8_13.0.png
alpha_value must be in range [3,6]
Threshold:  0.445839622656
Shot detected:  1

```


## shot_cut_edges

This code receives the paths for save the graphic, video and image of summarization, also the thresholds and the size of bins.

The count of shot detected are be displayed into console

To execute:

```
python3 shot_cut_edges.py video_in.mpg path_video_out path_graphic_out path_mosaic_out threshold
```

Example

```
→ python3 shot_cut_edges.py ../videos/lisa.mpg ../video_out/edge_difference ../graph_out/edge_difference ../resume_out/edge_difference 40

video name: ../video_out/edge_difference/lisa_pixel_difference_40.avi
graphic name: ../graph_out/edge_difference/lisa_pixel_difference_40.png
mosaic name: ../resume_out/edge_difference/lisa_pixel_difference_40.png
Shot detected:  3

```
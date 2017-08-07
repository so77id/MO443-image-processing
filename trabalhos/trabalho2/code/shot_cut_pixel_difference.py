import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

DEBUG=False
FILE_NAME="pixel_difference"
FRAMES_PER_SECOND=1
FRAMES_PER_ROW_IN_MOSAIC=5
WIDTH_FOR_MOSAIC=150

#######################################################################################################################
# Get file name
#######################################################################################################################
def get_filename(path, name, data, extension):
    filename = '{0}/{1}'.format(path, name)
    for d in data:
        filename = '{0}_{1}'.format(filename, d)
    filename = '{0}.{1}'.format(filename, extension)

    return filename


#######################################################################################################################
# Get video writer
#######################################################################################################################

def get_write_instance(video_out, cap_in):
    fps = FRAMES_PER_SECOND
    frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    width = cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    # Print video data

    if(DEBUG):
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : ",format(fps))
        print ("Frames : ",format(frames))
        print ("codec: ", codec)
        print ("Frame size: ", format((width, height)))

    return cv2.VideoWriter(video_out, int(codec), fps, (int(width),int(height)))


#######################################################################################################################
# Creating graphic
#######################################################################################################################
def create_graphic(graphic_out, graphic, threshold):
    threshold_line = np.ones(graphic.size)*threshold
    ranges = np.arange(graphic.size)

    plt.plot(ranges, graphic, 'b-', label="Video")
    plt.plot(ranges, threshold_line, 'r-', label="Threshold")

    plt.legend(loc='upper left', shadow=False)

    plt.ylabel('Metric')
    plt.xlabel('Frames')
    plt.grid(True)

    plt.savefig(graphic_out)
    if(DEBUG):
        plt.show()

#######################################################################################################################
# Creating mosaic from video resumen
#######################################################################################################################
def create_mosaic(mosaic_out, frames):
    height, width, channels = frames[0].shape
    new_width = WIDTH_FOR_MOSAIC
    new_height = int((height * new_width)/width)
    new_frame = None
    new_row = None
    new_img = None


    i = 0
    j = 0
    for frame in frames:
        new_frame = cv2.resize(frame,(new_width, new_height))
        if(i == 0):
            new_row = new_frame
        else:
            new_row = np.concatenate((new_row, new_frame), axis=1)

        i += 1
        if(i == FRAMES_PER_ROW_IN_MOSAIC):
            i = 0
            if(j == 0):
                new_img = new_row
            else:
                new_img = np.concatenate((new_img, new_row), axis=0)
            j+=1
    if(i < FRAMES_PER_ROW_IN_MOSAIC and i > 0):
        while(i < FRAMES_PER_ROW_IN_MOSAIC):
            i +=1
            new_frame = np.ones((new_height, new_width, channels))*255
            new_row = np.concatenate((new_row, new_frame), axis=1)
        if(j == 0):
            new_img = new_row
        else:
            new_img = np.concatenate((new_img, new_row), axis=0)

    if(DEBUG):
        plt.imshow(new_img)
        plt.show()
    cv2.imwrite(mosaic_out, new_img)


#######################################################################################################################
# Function that verified if differences of frames is a shot change
#######################################################################################################################
def shot_cut_pixel_differences(frame0, frame1, pixel_threshold, frame_threshold):
    diff = cv2.absdiff(frame0,frame1)
    diff[diff < pixel_threshold] = 0
    diff[diff >= pixel_threshold] = 1
    size = 1
    for x in diff.shape:
        size *= x
    ratio = diff.sum()/size
    if (ratio >= frame_threshold):
        return True, ratio
    else:
        return False, ratio



#######################################################################################################################
# Shot cut function
#######################################################################################################################

def shot_cut_video(video_in, video_out, graphic_out, mosaic_out, pixel_threshold, frame_threshold):
    # Variable to n and n+1 frame's
    graphic = np.array([])
    gray0 = gray1 = None
    frames = []

    # Get first frame
    if(video_in.isOpened()):
        ret, frame = video_in.read()
        if(ret == False):
            print("Video cannot opened")
            return -1

        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_out.write(frame)
        frames.append(frame)
        graphic = np.append(graphic, 0)

    count = 0
    while(video_in.isOpened()):
        # Get new frame
        ret, frame = video_in.read()
        # Break if this is the last frame
        if(ret == False):
            break

        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        change, value = shot_cut_pixel_differences(gray0, gray1, pixel_threshold, frame_threshold)

        if (change):
            #save frame
            if(DEBUG):
                print("cambio de frame")
            video_out.write(frame)
            frames.append(frame)
            count+=1

        graphic = np.append(graphic, value)

        # Next Frame
        gray0 = gray1

    print("Shot detected: ", count)
    create_graphic(graphic_out, graphic, frame_threshold)
    create_mosaic(mosaic_out, frames)


#######################################################################################################################
# MAINS function
#######################################################################################################################

def main(argv):
    if (len(argv) < 7):
        print("Incorrect number of arguments")
        print("Execute: python3 shot_cut_pixel_difference.py video_in path_video_out path_graphic_out path_mosaic_out pixel_threshold frame_threshold")
        return -1

    video_in = argv[1]
    path_video_out = argv[2]
    path_graphic_out = argv[3]
    path_mosaic_out = argv[4]
    pixel_threshold = int(argv[5])
    frame_threshold = int(argv[6])


    # Create file name from video out
    video_out = get_filename(path_video_out, video_in.split('/')[-1].split('.')[0], [FILE_NAME, pixel_threshold, frame_threshold], 'avi')
    # Create file name from graphic out
    graphic_out = get_filename(path_graphic_out, video_in.split('/')[-1].split('.')[0], [FILE_NAME, pixel_threshold, frame_threshold], 'png')
    # Create file name from graphic out
    mosaic_out = get_filename(path_mosaic_out, video_in.split('/')[-1].split('.')[0], [FILE_NAME, pixel_threshold, frame_threshold], 'png')

    print("video name:", video_out)
    print("graphic name:", graphic_out)
    print("mosaic name:", mosaic_out)

    # Data in range
    if(pixel_threshold < 0 or pixel_threshold > 256):
        print("pixel_threshold must be in range [0,256]")
        return -1

    if(frame_threshold < 0 or frame_threshold >= 100):
        print("frame_threshold must be in range [0,100]")
        return -1
    frame_threshold = frame_threshold/100

    # Get video data
    cap_in = cv2.VideoCapture(video_in)
    cap_out = get_write_instance(video_out, cap_in)

    # Call Shot cut function
    shot_cut_video(cap_in, cap_out, graphic_out, mosaic_out, pixel_threshold, frame_threshold)

    # Close file
    cap_in.release()
    cap_out.release()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
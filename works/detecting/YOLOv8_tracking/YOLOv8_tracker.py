import cv2
from ultralytics import YOLO

# to facilitate multi-streaming, we will be using
# threading.
import threading

# define the video file path
path = "~/dev/datasets/videos" # video file path
video_file1 = path + "/20231018/02_iphone6.MOV"
# video_file2 = 0

# load the YOLOv8 model as the backbone of object
# tracking system
# Two models to load: one for object detection, one for
# object segmetnation
model1=YOLO('/home/qiao/dev/giao/yolo_weights/yolov8_20240208/yolov8l_snowwork.pt')
# model2 = YOLO('~/dev/giao/yolo_weights/yolov8n.pt')

###########################################################
# target function for thread
def run_tracker_in_thread(filename, model, file_index):
    video = cv2.VideoCapture(filename)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.imshow("Tracking_Stream_" + str(file_index),
                   res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()

###################################################
# create the object tracking threads
tracker_thread1 = threading.Thread(target = run_tracker_in_thread,
                                  args = (video_file1, model1, 1),
                                  daemon=True)

# tracker_thread2 = threading.Tread(target = run_tracker_in_thread,
#                                   args = (video_file2,
#                                           model2, 2),
#                                   daemon=True)

# start the object tracking thread
tracker_thread1.start()
# tracker_thread2.start()

# thread handing and destroy windows
tracker_thread1.join()
# tracker_thread2.join()
cv2.destroyAllWindows()

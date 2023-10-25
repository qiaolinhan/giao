import cv2
import numpy as np 
import os 
import argparse 

# construct the argument parse and parse the argument 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "path of the video sequence")
args = vars(ap.parse_args())

# create mouse global coordinates 
x_mouse = 0 
y_mouse = 0 

# create thermal video fps variable (8 fps in this case) 
fps = 2

# mouse event function 
def mouse_events(event, x, y, flags, param):
    # mouse movement event 
    if event == cv2.EVENT_MOUSEMOVE:

        # update mouse global coordinates 
        global x_mouse 
        global y_mouse 

        x_mouse = x 
        y_mouse = y

# set up mouse events and prepare the thermal frame display
gray16_frame = cv2.imread("sample_ir.png", cv2.IMREAD_ANYDEPTH)
cv2.imshow('gray8', gray16_frame)
cv2.setMouseCallback('gray8', mouse_events) 

# loop over the thermal video frames 
for image in sorted(os.listdir(args["video"])):
    # filter .png (.tiff) files (gray16 images)
    if image.endswith(".png"):
        # define the gray16 frame path 
        file_path = os.path.join(args["video"], image)
        # open the gray16 frame 
        gray16_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        # calculate the tempreture
        temperature_pointer = gray16_frame[y_mouse, x_mouse]
#         temperature_pointer = (temperature_pointer / 100) -273.15 
        temperature_pointer = (temperature_pointer-32)/1.8

        # temperature_pointer = (temperature_pointer /
        # 100) * 9 / 5 - 459.67

        # conver tthe gray16 frame into gray8 
        gray8_frame = np.zeros((120, 160), dtype = np.uint8)
        gray8_frame = cv2.normalize(gray16_frame,
                                    gray8_frame, 0, 255,
                                    cv2.NORM_MINMAX)

        # qiao: need to be careful, it need to be uint8
        gray8_frame = np.uint8(gray8_frame)

        gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)
        # write the pointer 
        cv2.circle(gray8_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
        # write temperature 
        cv2.putText(img=gray8_frame,
                    text='{0:.1f}C'.format(temperature_pointer),
                    org=(x_mouse - 40, y_mouse - 40),
                    fontFace=0,
                    fontScale=2, color=(0, 0, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        # show the thermal frame 
        cv2.imshow("gray8", gray8_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # wait 125 ms: RWMVISION THERMALCAM1 fps = gray8_frame 
cv2.destroyAllWindows()     

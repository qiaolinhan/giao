import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, update_trace_list, draw_trace

# max iou as the evidence
initial_target_box = [729, 238, 764, 339]
if __name__ == "main":
    video_path = ""
    label_path = ""
    file_name = ""
    last_frame_box = initial_target_box
    cap = cv2.VideoCapture(video_path)
    frame_counter = 1
    cv2.nameWindow("track", cv2.WINDOW_NORMAL)
    trace_list = []

    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('boxoutput.avi', fourcc, 20, (768, 576))

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()

            for j, data_ in enumerate(content):
                data = data.replace('\n', "").split(" ")
                uvuv = np.array(data[1:5], dtype = "int")
                iou = cal_iou(uvuv, last_frame_box)

                plot_one_box(uvuv, frame)
                u1v1 = str(uvuv[0:2])
                u2v2 = str(uvuv[2:4])
                cv2.putText(frame, u1v1, (int(uvuv[0]), int(uvuv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, u2v2, (int(uvuv[2]), int(uvuv[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow('Track', frame)
        if SAVE_VIDEO:
            out.write(frame)
        frame_counter = frame_counter + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, uvuv2uvwh, uvwh2uvuv
initial_target_box = [729, 238, 764, 339]

initial_box_state = uvuv2uvwh(initial_target_box)
initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3], 0, 0, 0, 0]]).T

# trasfer matrix
F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])

# when the detection (measurement) missing
F_ = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
               [0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])
# observe matrix
H = np.eye(8)
# state noise, from the uncertainty as target moving
Q = np.eye(8) * 0.1
# observe noise, from the detection missing
R = np.eye(8) * 10
# control input
B = None
#convariance matrix of statet estimation, initial
P = np.eye(8)  

if __name__ == "__main__":

    video_path = ""
    label_path = ""
    file_name = ""
    cap = cv2.VideoCapture(video_path)
    # cv2.namedWindow("track", cv2.WINDOW_NORMAL)
    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('kalman_output.avi', fourcc, 20,(768,576))

# state initialization
frame_counter = 1
X_posterior = np.array(initial_state)
P_posterior = np.array(P)
Z = np.array(initial_state)

track_list = []

while (True):
    ret, frame = cap.read()
    last_box_posterior = uvwh2uvuv(X_posterior[0:4])
    plot_one_box(last_box_posterior, frame, color = (255, 255, 255), target = False)
    if not ret:
        break
    print(frame_counter)
    label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
    with open(label_file_path, "r") as f:
        content = f.readlines()
        max_iou = 0
        max_iou_matched = False
        for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="float")
                plot_one_box(xyxy, frame)
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                if iou > max_iou:
                    target_box = xyxy
                    max_iou = iou
                    max_iou_matched = True
            if max_iou_matched == True:
                # 如果找到了最大IOU BOX,则认为该框为观测值
                plot_one_box(target_box, frame, target=True)
                xywh = xyxy_to_xywh(target_box)
                box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
                trace_list = updata_trace_list(box_center, trace_list, 20)
                cv2.putText(frame, "Tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0), 2)
                # 计算dx,dy,dw,dh
                dx = xywh[0] - X_posterior[0]
                dy = xywh[1] - X_posterior[1]
                dw = xywh[2] - X_posterior[2]
                dh = xywh[3] - X_posterior[3]
                Z[0:4] = np.array([xywh]).T
                Z[4::] = np.array([dx, dy, dw, dh])

        if max_iou_matched:
            # -----进行先验估计-----------------
            X_prior = np.dot(A, X_posterior)
            box_prior = xywh_to_xyxy(X_prior[0:4])
            # plot_one_box(box_prior, frame, color=(0, 0, 0), target=False)
            # -----计算状态估计协方差矩阵P--------
            P_prior_1 = np.dot(A, P_posterior)
            P_prior = np.dot(P_prior_1, A.T) + Q
            # ------计算卡尔曼增益---------------------
            k1 = np.dot(P_prior, H.T)
            k2 = np.dot(np.dot(H, P_prior), H.T) + R
            K = np.dot(k1, np.linalg.inv(k2))
            # --------------后验估计------------
            X_posterior_1 = Z - np.dot(H, X_prior)
            X_posterior = X_prior + np.dot(K, X_posterior_1)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            # plot_one_box(box_posterior, frame, color=(255, 255, 255), target=False)
            # ---------更新状态估计协方差矩阵P-----
            P_posterior_1 = np.eye(8) - np.dot(K, H)
            P_posterior = np.dot(P_posterior_1, P_prior)
        else:
            # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
            # 此时直接迭代，不使用卡尔曼滤波
            X_posterior = np.dot(A_, X_posterior)
            # X_posterior = np.dot(A_, X_posterior)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            # plot_one_box(box_posterior, frame, color=(255, 255, 255), target=False)
            box_center = (
            (int(box_posterior[0] + box_posterior[2]) // 2), int((box_posterior[1] + box_posterior[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 20)
            cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

        draw_trace(frame, trace_list)


        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Last frame best estimation(White)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            out.write(frame)
        frame_counter = frame_counter + 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # 写完了
            
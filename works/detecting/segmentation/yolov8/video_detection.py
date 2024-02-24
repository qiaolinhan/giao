from ultralytics import YOLO
import numpy as np
import supervision as sv

Video_path = '/home/qiao/dev/datasets/videos/' + '20231016/forlocation.mp4'

model = YOLO('yolo_Weights/yolov8n150_snowwork.pt')

# video_info = sv.VideoInfo.from_video_path(Video_path)
#
# def process_frame(frame: np.ndarray, _) -> np.ndarray:
#     results = model(frame, imgsz=640)[0]
#     detections = sv.Detections.from_yolov8(results)
#     box_annotator = sv.BoxAnnotator(thickness=1,
#                                     text_thickness = 2,
#                                     text_scale=1)
#
#     labels = [f"{model.names[class_id]} {confidence:0.6}"
#               for _, _, confidence, class_id, _ in
#               detections]
#
#     frame = box_annotator.annotate(scene=frame,
#                                    detections=detections,
#                                    labels = labels)
#
#
# sv.process_video(source_path = Video_path, 
#                  target_path = f"result_yolov8l_20240116.mp4", 
#                  callback = process_frame)

result = model(Video_path, stream=True, save=True, exist_ok=True,
               name="result_20240208", conf=0.2)

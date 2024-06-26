import supervision as sv
from ultralytics import YOLO, RTDETR
import os
import torch
import time
import numpy as np
import cv2

video_path = 'C:\\Users\ADMIN\Desktop\AIC21-Track4-Anomaly-Detection_full\AIC21-Track4-Anomaly-Detection\\aic21-track4-train-data\\33.mp4'
# để cái file path của vid vào đây, nhớ chỉnh \ --> \\
video_info = sv.VideoInfo.from_video_path(video_path)
model_path = os.path.dirname('D:\\bi12year3\intern\ictlab\lmao\model\\')
model = YOLO('yolov8n.pt')
# model = RTDETR(model_path + '\\rtdetr-l.pt')

# torch.cuda.set_device(0)

fps= video_info.fps

thickness = sv.calculate_optimal_line_thickness(resolution_wh= video_info.resolution_wh)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness= thickness)

text_scale = sv.calculate_optimal_text_scale(resolution_wh= video_info.resolution_wh)

label_annotator = sv.LabelAnnotator(text_scale= text_scale, text_thickness= thickness, text_padding= thickness)


frame_generator = sv.get_video_frames_generator(video_path)

byte_track = sv.ByteTrack(frame_rate= fps, lost_track_buffer= 120)
frame_gen = sv.get_video_frames_generator(video_path)

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
classes = [0, 1, 2, 3, 5, 7]

def callback(frame: np.ndarray) -> sv.Detections:
    result = model.predict(frame, classes= classes, conf= 0.5, device= 'cuda:0')[0]
    return sv.Detections.from_ultralytics(result).with_nms(threshold= 0.25)

slicer = sv.InferenceSlicer(
    callback=callback,
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
)

smoother = sv.DetectionsSmoother()

for frame in frame_gen:
    # time when we finish processing for this frame 
    new_frame_time = time.time() 
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    fps2 = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 

    fps2 = int(fps2) 
    fps2 = str(fps2)


    result = model.predict(frame, classes= classes, conf= 0.5, device= 'cuda:0')[0]
    detections = sv.Detections.from_ultralytics(result)

    # detections = slicer(frame)
    detections = byte_track.update_with_detections(detections= detections)
    # detections = smoother.update_with_detections(detections)

    labels = [
                # f"{tracker_id} x:{x} y:{y}"
                f'{tracker_id}'
                # f"{model.names[class_id]}:{tracker_id} x:{x} y:{y}"
                for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
                ]
 
    annotated_frame = frame.copy()

    annotated_frame = bounding_box_annotator.annotate(
        scene= annotated_frame,
        detections= detections,
        custom_color_lookup= None
    )   

    annotated_frame = label_annotator.annotate(
                    scene= annotated_frame,
                    detections= detections,
                    labels= labels,
                    )
    
    cv2.putText(annotated_frame, fps2, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
            
    cv2.imshow(f"annotated_frame", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
    else:
        pass

cv2.destroyAllWindows()

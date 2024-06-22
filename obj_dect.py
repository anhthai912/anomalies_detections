import supervision as sv
from ultralytics import YOLO, RTDETR
import cv2
import os
from collections import OrderedDict
import torch
import numpy as np
import time
from mod import CustomBoundingBoxAnnotator, check_life, add_check, CONFIG
from general import PATHS

# PATHS = {
#     'general': "D:\\bi12year3\intern\ictlab\imgonnacrylmao\\", #folder path
#     'dataset' : "C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\", #dataset path
# }

# PATHS = {
#     'general' : None,
#     'model' : None
# }
# DATASET_PATH = os.path.dirname("C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\")
# MODEL_PATH = os.path.dirname('D:\\bi12year3\intern\ictlab\lmao\model\\')


def run_main(vid_no, mode: str = "train"):
    if mode == "train":
        print("*************************************************\nmode: TRAIN\n**********************************************\n")
        folder_path = PATHS["dataset"] + '\\aic21-track4-train-data\\'
        result_path = PATHS["general"] + f'results_train\\Train_Output_{vid_no}_anomalies.txt'
    elif mode == "test":
        print("############################################\nmode:test\n#######################################################\n")
        folder_path = PATHS["dataset"] + '\\aic21-track4-test-data\\'
        result_path = PATHS["general"] + f'results_test\\Test_Output_{vid_no}_anomalies.txt'
    else:
        folder_path = None
        result_path = None

    try:
        video_path = folder_path + f'\\{vid_no}.mp4'
        video_info = sv.VideoInfo.from_video_path(video_path= video_path)
    
    except:
        print("#####################\nICORRECT MODE\n**************************")

    try:
        torch.cuda.set_device(0) # Set to your desired GPU number
    except:
        print("\nNo cuda or incorrectly installed\n")

    # model = RTDETR(model_path + '\\rtdetr-l.pt')
    model = YOLO(PATHS['general'] + '\\models\\yolov8n.pt')

    fps = video_info.fps

    frame_generator = sv.get_video_frames_generator(video_path)

    # tracking
    byte_track = sv.ByteTrack(frame_rate= fps, lost_track_buffer= 120)

    text_file = open(result_path, 'w') 

    vehicle_id = OrderedDict()

    anomalies = {}
    frame_no = 0

    start = time.time()
    for frame in frame_generator:
        
        timer = frame_no/fps
        timer = int(round(timer, 0))
        print(f"vid no: {vid_no}")
        print(frame_no)
        print(timer)
        if frame_no >= 0:
        # if timer > 0:
            vid_time = [frame_no, timer]
            
            result = model.predict(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections= detections)
            if frame_no % CONFIG['frame_skip'] == 0:
                vehicle_id = add_check(detections, vehicle_id, anomalies, vid_time)

            anomalies, vehicle_id = check_life(vehicle_id, anomalies, vid_time)
            print(anomalies)
            frame_no += 1

        else:
            pass
    print(anomalies)

    end = time.time()
    timer = end - start
    timer = round(timer, 2) 


    for i in anomalies.items():
        text_file.write(f"{str(i)}\n")

    text_file.write(f"\nStart to end:{timer}")
    text_file.write(f"\nnumber of anomalies {str(len(anomalies))}")

    text_file.close()
    print('done')



def run_main_img(vid_no, mode: str = "train"):
    if mode == "train":
        print("*************************************************\nmode: TRAIN\n**********************************************\n")
        folder_path = PATHS["dataset"] + '\\aic21-track4-train-data\\'
        result_path = PATHS["general"] + f'results_train\\Train_Output_{vid_no}_anomalies.txt'
    elif mode == "test":
        print("############################################\nmode:test\n#######################################################\n")
        folder_path = PATHS["dataset"] + '\\aic21-track4-test-data\\'
        result_path = PATHS["general"] + f'results_test\\Test_Output_{vid_no}_anomalies.txt'
    else:
        folder_path = None
        result_path = None

    try:
        video_path = folder_path + f'\\{vid_no}.mp4'
        video_info = sv.VideoInfo.from_video_path(video_path= video_path)
    
    except:
        print("#####################\nICORRECT MODE\n**************************")

    try:
        torch.cuda.set_device(0) # Set to your desired GPU number
    except:
        print("\nNo cuda or incorrectly installed\n")

    # model = RTDETR(model_path + '\\rtdetr-l.pt')
    model = YOLO(PATHS['general'] + '\\models\\yolov8n.pt')

    fps = video_info.fps

    thickness = sv.calculate_optimal_line_thickness(resolution_wh= video_info.resolution_wh)
    bounding_box_annotator = CustomBoundingBoxAnnotator(thickness= thickness)

    text_scale = sv.calculate_optimal_text_scale(resolution_wh= video_info.resolution_wh)
    # text_scale = 2

    label_annotator = sv.LabelAnnotator(text_scale= text_scale, text_thickness= thickness, text_padding= thickness)


    frame_generator = sv.get_video_frames_generator(video_path)

    # tracking
    byte_track = sv.ByteTrack(frame_rate= fps, lost_track_buffer= 120)

    # if mode == "train": 
    #     text_file = open(PATHS["general"] + f'results_train\\Train_Output_{vid_no}_anomalies.txt', "w")
    # elif mode == "test":
    #     text_file = open(PATHS["general"] + f'results_test\\Test_Output_{vid_no}_anomalies.txt', 'w')
    # else:
    #     try:
    #         text_file = open("dfds", "sadfds", 00, "sdfsdf") 
    #     except:
    #         print('\n Incorrect Mode \n')

    text_file = open(result_path, 'w') 

    vehicle_id = OrderedDict()

    anomalies = {}
    frame_no = 0

    start = time.time()
    for frame in frame_generator:
        
        timer = frame_no/fps
        timer = int(round(timer, 0))
        print(f"vid no: {vid_no}")
        print(frame_no)
        print(timer)
        if frame_no >= 0:
        # if timer > 0:
            vid_time = [frame_no, timer]
            
            result = model.predict(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections= detections)
            if frame_no % CONFIG['frame_skip'] == 0:
                vehicle_id = add_check(detections, vehicle_id, anomalies, vid_time)

            anomalies, vehicle_id = check_life(vehicle_id, anomalies, vid_time)
            print(anomalies)
            frame_no += 1

# TODO: fix the value error: tracking id -> emty is the problem(maybe)

               
            # labels = [
            #     # f"{tracker_id} x:{x} y:{y}"
            #     f'{tracker_id}'
            #     # f"{model.names[class_id]}:{tracker_id} x:{x} y:{y}"
            #     for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
            #     ]
 
            annotated_frame = frame.copy()

            annotated_frame = bounding_box_annotator.annotate(
                scene= annotated_frame,
                detections= detections,
                anomalies= anomalies,
                custom_color_lookup= None
            )   
            
            # try: 
            #     annotated_frame = label_annotator.annotate(
            #         scene= annotated_frame,
            #         detections= detections,
            #         labels= labels,
            #         )
            # except ValueError as e:
            #     print("ERROR", e)
            #     print("DETECTION: ", detections)
            #     text_file.write("\n*******************************************\n")
            #     text_file.write(f"\nERROR: {e}\n")
            #     text_file.write(f"\nDETECTIONS: {detections}\n")
            #     text_file.write("\n*******************************************\n")

            annotated_frame = cv2.putText(
                annotated_frame, f"frame: {frame_no} \\time: {timer}", 
                org= (20, 50), 
                fontFace= cv2.FONT_HERSHEY_COMPLEX,
                fontScale= 1, 
                color= (0, 255, 0), 
                thickness= 2)
            
            cv2.imshow(f"annotated_frame{vid_no}", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            pass

    cv2.destroyAllWindows()

    print(anomalies)

    end = time.time()
    timer = end - start
    timer = round(timer, 2) 


    for i in anomalies.items():
        text_file.write(f"{str(i)}\n")

    text_file.write(f"\nStart to end:{timer}")
    text_file.write(f"\nnumber of anomalies {str(len(anomalies))}")

    text_file.close()
    print('done')

# check_dir()
run_main_img(33, "train")
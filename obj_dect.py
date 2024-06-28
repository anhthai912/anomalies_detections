import supervision as sv
from ultralytics import YOLO, RTDETR
import cv2
import os
from collections import OrderedDict
import torch
import numpy as np
import time
from mod import CustomBoundingBoxAnnotator, check_life, add_check, CONFIG, PATHS
from boxmot import BYTETracker, OCSORT, BoTSORT, HybridSORT
import progressbar
from pathlib import Path


# PATHS = {
#     'general': "D:\\bi12year3\intern\ictlab\imgonnacrylmao\\", #folder path
#     'dataset' : "C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\", #dataset path
# }

# DATASET_PATH = os.path.dirname("C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\")
# MODEL_PATH = os.path.dirname('D:\\bi12year3\intern\ictlab\lmao\model\\')

classes = [0, 1, 2, 3, 5, 7]

model = YOLO(PATHS['general'] + '\\models\\yolov8n.pt')


def run_main(vid_no, mode: str = "train", confident= 0.5, show= False, save= True,
             frame_skip= CONFIG['frame_skip'], 
             errors= CONFIG['error'], 
             life_time= CONFIG['life_time'], 
             tracker_memo= CONFIG['tracker_memo'],
             tracker_n= 'bytetrack'):
    
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
        print(folder_path + f'\\{vid_no}.mp4')
        print("#####################\nICORRECT MODE\n**************************")

    try:
        torch.cuda.set_device(0) # Set to your desired GPU number
    except:
        print("\nNo cuda or incorrectly installed\n")

    # model = RTDETR(model_path + '\\rtdetr-l.pt')
    # model = YOLO('yolov8n.pt')

    if tracker_n == 'bytetrack': 
        tracker = BYTETracker(
            track_thresh= 0.25,
            track_buffer= 120)

    elif tracker_n == 'ocsort':
        tracker = OCSORT()  

    elif tracker_n == 'hybridsort':
        tracker = HybridSORT(
            reid_weights= Path('osnet_x0_25_msmt17.pt'),
            device= 0,
            half= True,
            det_thresh= 0.5
        )
    elif tracker_n == 'botsort':
        tracker = BoTSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'),
            device= 0,
            fp16= 1,
        )
    else: 
        tracker = None
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NO TRACKER SELECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # cuda0 = torch.device('cuda:0')
    # model.to(cuda0)

    fps = video_info.fps

    if show:
        thickness = sv.calculate_optimal_line_thickness(resolution_wh= video_info.resolution_wh)
        bounding_box_annotator = CustomBoundingBoxAnnotator(thickness= thickness)

        # text_scale = sv.calculate_optimal_text_scale(resolution_wh= video_info.resolution_wh)
        # text_scale = 2

        # label_annotator = sv.LabelAnnotator(text_scale= text_scale, text_thickness= thickness, text_padding= thickness)


    frame_generator = sv.get_video_frames_generator(video_path)

    # tracking
    # byte_track = sv.ByteTrack(frame_rate= fps, lost_track_buffer= 120)

    smother = sv.DetectionsSmoother()
    
    vehicle_id = OrderedDict()

    anomalies = {}
    frame_no = 0

    total_frame = video_info.total_frames

    print(f"vid no: {vid_no}")

    
    with progressbar.ProgressBar(max_value=total_frame) as bar:
        start = time.time()
        bar.update(frame_no)
        for frame in frame_generator:
            bar.update(frame_no)
            timer = frame_no/fps
            timer = int(round(timer, 0))
            # print(f"vid no: {vid_no}")
            # print(frame_no)
            # print(timer)
            if frame_no >= 0:
            # if timer > 0:
                vid_time = [frame_no, timer]
                
                result = model.predict(frame, classes= classes, conf= confident, device= 'cuda:0', verbose = False)[0]
                detections = sv.Detections.from_ultralytics(result).with_nms(threshold= CONFIG['nms'])
                
                if detections is None or len(detections.xyxy) == 0:
                    tracker_output = np.empty((0, 8))
                else:
                    dect_xyxy = detections.xyxy
                    dect_conf = detections.confidence
                    dect_cls = detections.class_id

                    dect = []
                    for i in range(len(dect_xyxy)):
                        val = dect_xyxy[i].tolist()
                        val.append(dect_conf[i])
                        val.append(dect_cls[i])
                        dect.append(val)

                    dect = np.array(dect)
                    # print("hello")
                    # print(dect)
                    # print(len(dect.shape))
                    # print(detections)
                    tracker_output = tracker.update(dect, frame) # --> M X (x, y, x, y, id, conf, cls, ind)

                # print()
                # print(tracker_output)
                # tracker.plot_results(frame, show_trajectories= False)
                if tracker_output.size == 0:
                    detections = sv.Detections(
                        xyxy=np.empty((0, 4)), 
                        confidence=np.empty((0,)), 
                        class_id=np.empty((0,), dtype=int), 
                        tracker_id=np.empty((0,))
                    )
                else:
                    detections = sv.Detections(
                        xyxy=tracker_output[:, :4], 
                        confidence=tracker_output[:, 5], 
                        class_id=tracker_output[:, 6].astype(int), 
                        tracker_id=tracker_output[:, 4]
                    )
                detections = smother.update_with_detections(detections)
                # detections = Tracker_info(detections)
                
                if frame_no % frame_skip == 0:#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    vehicle_id = add_check(detections, vehicle_id, anomalies, vid_time, tracker_memo, errors)

                anomalies, vehicle_id = check_life(vehicle_id, anomalies, vid_time, life_time)
                # print(anomalies)
                frame_no += 1

                if show:
                    # TODO: fix the value error: label

                    # labels = [
                    #     # f"{tracker_id} x:{x} y:{y}"
                    #     f'{tracker_id}'
                    #     # f"{model.names[class_id]}:{tracker_id} x:{x} y:{y}"
                    #     for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
                    #     ]
        
                    annotated_frame = frame.copy()
                    # if len(tracker_output) != 0:
                        
                    annotated_frame = bounding_box_annotator.annotate(
                        scene= annotated_frame,
                        detections= detections,
                        anomalies= anomalies,
                        custom_color_lookup= None
                    )   

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
        if show:
            cv2.destroyAllWindows()

    # bar.finish()


    # print(anomalies)

    if save == True:
        end = time.time()
        timer = end - start
        timer = round(timer, 2) 

        text_file = open(result_path, 'w')
        for i in anomalies.items():
            text_file.write(f"{str(i)}\n")

        text_file.write(f"\nStart to end:{timer}")
        text_file.write(f"\nnumber of anomalies {str(len(anomalies))}")

        text_file.close()
        print(f'done {vid_no}')
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!     WARNING   WARNING    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"{vid_no} not saved !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pass

if __name__ == '__main__':
    # run_main(9, "train", show= True, save= True,frame_skip= 4,
    #          life_time= 30,
    #          errors= 1, tracker_n= 'bytetrack')































def run_main_old(vid_no, mode: str = "train", confident= 0.5, show: bool= False, save: bool= True,
             frame_skip= CONFIG['frame_skip'], 
             errors= CONFIG['error'], 
             life_time= CONFIG['life_time'], 
             tracker_memo= CONFIG['tracker_memo']):
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
        print(folder_path + f'\\{vid_no}.mp4')
        print("#####################\nICORRECT MODE\n**************************")

    try:
        torch.cuda.set_device(0) # Set to your desired GPU number
    except:
        print("\nNo cuda or incorrectly installed\n")

    # model = RTDETR(model_path + '\\rtdetr-l.pt')
    
    # cuda0 = torch.device('cuda:0')
    # model.to(cuda0)

    fps = video_info.fps

    if show:
        thickness = sv.calculate_optimal_line_thickness(resolution_wh= video_info.resolution_wh)
        bounding_box_annotator = CustomBoundingBoxAnnotator(thickness= thickness)

        text_scale = sv.calculate_optimal_text_scale(resolution_wh= video_info.resolution_wh)
        # text_scale = 2

        label_annotator = sv.LabelAnnotator(text_scale= text_scale, text_thickness= thickness, text_padding= thickness)


    frame_generator = sv.get_video_frames_generator(video_path)

    # tracking
    byte_track = sv.ByteTrack(frame_rate= fps, lost_track_buffer= 120)

    smother = sv.DetectionsSmoother()
    
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
            
            result = model.predict(frame, classes= classes, conf= confident, device= 'cuda:0')[0]
            detections = sv.Detections.from_ultralytics(result).with_nms(threshold= CONFIG['nms'])
            detections = byte_track.update_with_detections(detections= detections)
            detections = smother.update_with_detections(detections)
            if frame_no % frame_skip == 0:#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                vehicle_id = add_check(detections, vehicle_id, anomalies, vid_time, tracker_memo, errors)

            anomalies, vehicle_id = check_life(vehicle_id, anomalies, vid_time, life_time)
            print(anomalies)
            frame_no += 1

            if show:
                # TODO: fix the value error: label

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
    if show:
        cv2.destroyAllWindows()

    print(anomalies)

    if save == True:
        end = time.time()
        timer = end - start
        timer = round(timer, 2) 

        text_file = open(result_path, 'w')
        for i in anomalies.items():
            text_file.write(f"{str(i)}\n")

        text_file.write(f"\nStart to end:{timer}")
        text_file.write(f"\nnumber of anomalies {str(len(anomalies))}")

        text_file.close()
        print('done')
    else:
        pass

# check_dir()
# for i in range(1, 101):
# run_main(33, "train", show= True, save= True,frame_skip= 3,
#          life_time= 60,
#          errors= 1,)



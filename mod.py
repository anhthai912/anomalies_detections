from typing import List, Optional, Union
from math import isclose
import cv2
import numpy as np
import supervision as sv
import os

from general import PATHS
from supervision.annotators.base import ImageType
from supervision.annotators.utils import (
    ColorLookup,
    get_color_by_index,
    resolve_color_idx,
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


CONFIG = {
    'error' : 1,
    'life_time' : 30,
    'tracker_memo' : 1000,
    'frame_skip' : 2
}

class CustomBoundingBoxAnnotator(sv.BoundingBoxAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        super().__init__(
            color, thickness, color_lookup
        )
    
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        anomalies: dict,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        # detections_copy = detections.copy()
        for detection_idx in range(len(detections)):
            x1,y1,x2,y2 = detections.xyxy[detection_idx].astype(int)

            temp = None
            try:
                temp = detections.tracker_id[detection_idx]
            except IndexError as e:
                print("error at CustomBoundingBox class")
                print("DETECTION: ",detections)
                print("TRACKER_IDS",detections.tracker_id)
                print("ERROR: ", e)

            color = custom_resolve_color(
                color= self.color,
                detections= detections,
                detection_idx= detection_idx,
                track_idx= temp,
                anomalies= anomalies,
                color_lookup= self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            
            cv2.rectangle(
                img= scene,
                pt1= (x1, y1),
                pt2= (x2, y2),
                color= color.as_bgr(),
                thickness= self.thickness
            )
            

        return scene

def custom_resolve_color(
    color: Union[Color, ColorPalette],
    detections: Detections,
    detection_idx: int,
    anomalies: dict,
    track_idx: int = None,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> Color:
    idx = resolve_color_idx(
            detections= detections,
            detection_idx= detection_idx,
            color_lookup= color_lookup,
        )
    for ano_id in anomalies.keys():
        if ano_id == track_idx:
            print(ano_id)
            return sv.Color.RED
    return get_color_by_index(color= color, idx= idx)

def coor_check(old_coor, new_coor, errors = CONFIG["error"]):
    check = 0
    for i in range(len(old_coor)):
        if isclose(old_coor[i], new_coor[i], abs_tol= errors):
            check += 1
    return check == 4

def update_checker(tracker_id: dict, Id, new_coor, new_start):
    old_coor, life, old_start, end = tracker_id[Id]
    if coor_check(old_coor, new_coor):
        life += 1
        tracker_id.update({Id: [new_coor, life, old_start, end]})
        return tracker_id
    else:
        life = 0
        tracker_id[Id] = [new_coor, life, new_start, end]
        return tracker_id
    
def add_check(detections: Detections, vehicle_id: dict, anomalies: dict, vid_time: list):
    vehi_cop = vehicle_id.copy()
    for tracker_id, coors in zip(detections.tracker_id, detections.xyxy):
        coors = list(coors)
        for i in range(len(coors)):
            coors[i] = int(coors[i])
        if tracker_id not in vehicle_id:
            vehi_cop[tracker_id]= [coors, 0, vid_time, vid_time]
            if len(vehicle_id.items()) >= CONFIG["tracker_memo"]: #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                temp = vehi_cop.popitem(last= False)
                if temp[0] in anomalies.keys():
                    vehi_cop[temp[0]] = temp[1]
        else:
            vehi_cop = update_checker(vehi_cop, tracker_id, coors, vid_time)
    return vehi_cop

def add_anomaly(vehicle, anomalies: dict, ano_time):
    ano = anomalies.copy()
    # track = track_ids.copy()
    coors, life, false_start, ano_end = vehicle[1]
    if vehicle[0] not in ano.keys():
        # ano[vehicle[0]] = [x, y, life, vehicle[1][3], ano_end]
        ano[vehicle[0]] = vehicle[1]
    else:
        ano[vehicle[0]][:2] = coors ,life
        ano[vehicle[0]][-1] = ano_time
        
    return ano

def check_life(tracker_id:dict, anomalies: dict, ano_time):
    for life in tracker_id.items():
        if int(life[1][1]) > CONFIG["life_time"]:
            anomalies = add_anomaly(life, anomalies, ano_time)
    return anomalies, tracker_id

# #############################################################

def read_anomalies(vid_no, mode):
#   data_path = os.path.dirname(data_path)
    if mode == "train":
        file_path = PATHS['general'] + f"results_train\\Train_Output_{vid_no}_anomalies.txt"
    elif mode == "test":
        file_path = PATHS["general"] + f'results_test\\Test_Output_{vid_no}_anomalies.txt'
    else:
        file_path = None

    try:
        text_file = open(file_path, "r")
    except:
        print(file_path)
        print("#####################\nICORRECT MODE\n**************************")

    lines = text_file.readlines()
    no_anomalies = int(lines[-1][20:])

    ano = {}

    for i in range(no_anomalies):
        line = lines[i]
        line = line.strip().strip('()')
        key, value = line.split(',', 1)
        key = int(key.strip())
        value = eval(value.strip())
        
        # Assign the key-value pair to the dictionary
        ano[key] = value
    return ano


def sort_ano(anomailes: dict):
    ano = {}
    anomailes_cop = anomailes.copy()

    for ano_id1 in anomailes.keys():
        if ano_id1 in anomailes_cop.keys():
            coors1 = anomailes[ano_id1][0]
            start1 = anomailes[ano_id1][-2]
            end1 = anomailes[ano_id1][-1]
            ano[ano_id1] = [coors1, start1, end1]
            temp = []
            for ano_id2 in anomailes_cop.keys():
                if ano_id1 == ano_id2:
                    pass
                else:
                    coors2 = anomailes[ano_id2][0]
                    start2 = anomailes[ano_id2][-2]
                    end2 = anomailes[ano_id2][-1]

                    if coor_check(coors1, coors2, errors= CONFIG["error"] + 2):
                        ano[ano_id1] = [coors1, start1, end2]
                        temp.append(ano_id2)
            for i in temp:
                anomailes_cop.pop(i)  
    return ano


def get_ano_info(anomailes: dict, id):
    # print(anomailes[id])
    coors = anomailes[id][0]
    start = anomailes[id][-2]
    end = anomailes[id][-1]
    return coors, start, end


def display_ano(vid_no, ano_data, mode): 
    coors, start, end = ano_data

    if mode == "train":
        file_path = PATHS["dataset"] + f'\\aic21-track4-train-data\\{vid_no}.mp4'
    elif mode == "test":
        file_path = PATHS["dataset"] + f'\\aic21-track4-test-data\\{vid_no}.mp4'
    else:
        file_path = None

    try:
        cap = cv2.VideoCapture(file_path)
    except:
        print("#####################\nICORRECT MODE\n**************************")

    # cap = cv2.VideoCapture(f"C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\aic21-track4-train-data\\{vid_no}.mp4")
    x1, x2, y1, y2 = coors
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    fps = 30
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start[0])
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if current_frame > end[0]:
            break

        timer = int(round(current_frame/fps, 0))

        if ret == True:
            # Display the resulting frame
            frame = cv2.rectangle(frame, 
                                pt1=(x1, x2), 
                                pt2= (y1, y2),  
                                color= (0, 0, 255), thickness= 2)
            
            frame = cv2.putText(frame, f'time: {timer}', 
                          org= (20, 50), 
                          fontFace= cv2.FONT_HERSHEY_COMPLEX,
                          fontScale= 1, 
                          color= (0, 255, 0), 
                          thickness= 1)

            cv2.imshow(f'frame{vid_no}',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        # else: 
        #   break


    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def process_ano(vid_no, mode):
    return sort_ano(anomailes= read_anomalies(vid_no, mode))

def play_ano(vid_no, mode, sorted_ano: dict):
    print("Number of anomalies: ", len(sorted_ano.keys()) )
    for i, ano_id in enumerate(sorted_ano.items()):
        print(f"Anomaly {i+1}")
        print(ano_id)
        display_ano(vid_no, get_ano_info(sorted_ano, ano_id[0]), mode)

# play_ano(vid_no, process_ano(vid_no))
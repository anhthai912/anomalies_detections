from typing import Optional, Union
# from math import isclose
import cv2
import numpy as np
import supervision as sv
import os


from supervision.annotators.base import ImageType
from supervision.annotators.utils import (
    ColorLookup,
    get_color_by_index,
    resolve_color_idx,
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

PATHS = {
    'general': "D:\\bi12year3\intern\ictlab\imgonnacrylmao\\", #folder path
    'dataset' : "C:\\Users\\ADMIN\\Desktop\\AIC21-Track4-Anomaly-Detection_full\\AIC21-Track4-Anomaly-Detection\\", #dataset path
    'read_slave' : "D:\\bi12year3\intern\gpu_slaves\\"
}

CONFIG = {
    'error' : 1,
    'life_time' : 30,
    'frame_skip' : 2,
    'anomaly_range': 20,
    'anomaly_min_time': 30,
    'nms': 0.25,
    'tracker_memo' : 1000,
}

def check_dir(paths = PATHS["general"]):
    newpath = [
    paths + '\\results',
    paths + '\\results_train',
    paths + '\\results_test',
    paths + '\\output'
    ] 
    for i in range(len(newpath)):
        if not os.path.exists(newpath[i]):
            os.makedirs(newpath[i])
            print(newpath[i])
check_dir()

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
            # print(ano_id)
            return sv.Color.RED
    return get_color_by_index(color= color, idx= idx)

def coor_check(old_coor, new_coor, errors = CONFIG["error"]):
    return all(abs(c1 - c2) <= errors for c1, c2 in zip(old_coor, new_coor))

def update_checker(tracker_id: dict, Id, new_coor, new_start, errors = CONFIG["error"]):
    old_coor, life, old_start, end = tracker_id[Id]
    if coor_check(old_coor, new_coor, errors):#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        life += 1
        tracker_id.update({Id: [new_coor, life, new_start, end]})
        return tracker_id
    else:
        life = 0
        tracker_id[Id] = [new_coor, life, new_start, end]
        return tracker_id
    
def add_check(detections: Detections, vehicle_id: dict, anomalies: dict, vid_time: list, tracker_memo= CONFIG['tracker_memo'], errors = CONFIG["error"]):
    vehi_cop = vehicle_id.copy()
    for tracker_id, coors in zip(detections.tracker_id, detections.xyxy):
        coors = list(coors)
        for i in range(len(coors)):
            coors[i] = int(coors[i])
        if tracker_id not in vehicle_id:
            vehi_cop[int(tracker_id)]= [coors, 0, vid_time, vid_time]
            if len(vehicle_id.items()) >= tracker_memo: #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                temp = vehi_cop.popitem(last= False)
                if temp[0] in anomalies.keys():
                    vehi_cop[temp[0]] = temp[1]
        else:
            vehi_cop = update_checker(vehi_cop, tracker_id, coors, vid_time, errors)#<<<<<<<<<<<<<<<<<<<<
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

def check_life(tracker_id:dict, anomalies: dict, ano_time, life_time = CONFIG['life_time']):
    for life in tracker_id.items():
        if int(life[1][1]) > life_time:#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            anomalies = add_anomaly(life, anomalies, ano_time)
    return anomalies, tracker_id

# #############################################################

def read_anomalies(vid_no, mode, paths):
#   data_path = os.path.dirname(data_path)
    if mode == "train":
        file_path = paths + f"results_train\\Train_Output_{vid_no}_anomalies.txt"
    elif mode == "test":
        file_path = paths + f'results_test\\Test_Output_{vid_no}_anomalies.txt'
    else:
        file_path = None

    try:
        text_file = open(file_path, "r")
    except:
        print(file_path)
        print("#####################\nICORRECT MODE\n**************************")

    lines = text_file.readlines()
    try:
        no_anomalies = int(lines[-1][20:])
    except:
        print()
        print(vid_no)
        print("\nvid_empty\n")
    ano = {}

    for i in range(no_anomalies):
        line = lines[i]
        line = line.strip().strip('()')
        key, value = line.split(',', 1)
        key = int(key.strip())
        value = eval(value.strip())
        
        # Assign the key-value pair to the dictionary
        ano[key] = value

    text_file.close()

    return ano


def sort_ano(anomalies: dict, anomaly_range= CONFIG['anomaly_range'], anomaly_min_time= CONFIG['anomaly_min_time']):
    # filtered_data = {k: v for k, v in anomalies.items() if (v[3][1] - v[2][1]) >= anomaly_min_time}
    filtered_data = anomalies.copy()
    merged_data = {}
    visited_keys = set()

    for key1, value1 in filtered_data.items():
        if key1 in visited_keys:
            continue

        coords1, trash1, start1, end1 = value1
        merged_start, merged_end = start1, end1

        for key2, value2 in filtered_data.items():
            if key2 <= key1 or key2 in visited_keys:
                continue

            coords2, trash2, start2, end2 = value2
            if coor_check(coords1, coords2, errors= anomaly_range):
                merged_start = min(merged_start, start2)
                merged_end = max(merged_end, end2)
                visited_keys.add(key2)

        merged_data[key1] = [coords1, merged_start, merged_end]
        visited_keys.add(key1)
    
    # return merged_data

    filtered_data_2 = {k: v for k, v in merged_data.items() if (v[2][1] - v[1][1]) >= anomaly_min_time}
    return filtered_data_2


def get_ano_info(anomailes: dict, id):
    # print(anomailes[id])
    coors = anomailes[id][0]
    start = anomailes[id][-2]
    end = anomailes[id][-1]
    return coors, start, end


def display_ano(vid_no, ano_data, mode, anomaly_range= CONFIG['anomaly_range']): 
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
                                pt1=(x1 - anomaly_range, x2 - anomaly_range), 
                                pt2= (y1 + anomaly_range, y2 + anomaly_range),  
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
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def process_ano(vid_no, mode, paths = PATHS['general'], 
                anomaly_range= CONFIG['anomaly_range'], 
                anomaly_min_time = CONFIG['anomaly_min_time']):
    return sort_ano(anomalies= read_anomalies(vid_no, mode, paths), 
                    anomaly_range= anomaly_range, 
                    anomaly_min_time= anomaly_min_time)

def play_ano(vid_no, mode, sorted_ano: dict, anomaly_range = CONFIG['anomaly_range']):
    print("Number of anomalies: ", len(sorted_ano.keys()) )
    for i, ano_id in enumerate(sorted_ano.items()):
        print(f"Anomaly {i+1}")
        print(ano_id)
        display_ano(vid_no, get_ano_info(sorted_ano, ano_id[0]), mode, anomaly_range= anomaly_range)

# play_ano(vid_no, process_ano(vid_no))
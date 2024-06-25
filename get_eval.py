from mod import process_ano, read_anomalies, sort_ano, PATHS, CONFIG
import os
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

TRUE_VID = [2, 9, 11, 14, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
# print(len(TRUE_VID))

def read_prediction_data(main_path, mode, select= []):
    result_path = f"{main_path}\\results_{mode}"
    predict = {}
    # Iterate directory
    for path in os.listdir(result_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(result_path, path)):
            key = int(path.split('_', 4)[2])
            if len(select) != 0 and key not in select:
                pass
            else:
                result_mini = read_anomalies(key, mode, paths= main_path)
                predict[key] = result_mini

    return predict

def get_prediction_data(prediction_dict: dict, 
                        anomaly_range= CONFIG['anomaly_range'], 
                        anomaly_min_time = CONFIG['anomaly_min_time']):
    predict = []

    for i in prediction_dict.items():
        key = i[0]
        result_mini = sort_ano(anomalies= i[1], 
                        anomaly_range= anomaly_range, 
                        anomaly_min_time= anomaly_min_time)
        if len(result_mini.keys()) != 0:
            for ano_i in result_mini.values():
                start, end = ano_i[-2][1], ano_i[-1][1]
                predict.append([key, start, end])
        else:
            predict.append([key, 0, 0])

    return sorted(predict)
    
def get_true_data(dataset_path):
    path = dataset_path + "\\train-anomaly-results.txt"
    evaluation = []

    eval_file = open(path, 'r')
    for i in eval_file:
        vid, start, end = i.split(' ', 2)
        end = end.strip('\n')
        vid,start, end = int(vid),int(start), int(end)
        evaluation.append([vid,start,end])
    eval_file.close()

    return sorted(evaluation)

def kms(lists):
    temp_list = lists.copy()
    temp_list.append([99999, 0, 0])
    keys = {}
    count = 1
    for i in range(len(lists)):
        # print("hello", lists[i][0])
        # print("AAAAAAAAAAAAA", temp_list[i+1][0])
        if lists[i][0] == temp_list[i+1][0]:
            
            count += 1
            # print("hello????", lists[i][0])
            keys[lists[i][0]] = count
        else:
            keys[lists[i][0]] = count
            count = 1
        
    return keys

def matching_result(y_pre, y_true, keys_true):
    new_true = []
    keys_true = kms(y_true)
    keys_predict = kms(y_pre)
    keys_add = {}

    temp_y_true = y_true.copy()
    temp_y_true.append([99999, 0, 0])
    # print("**************", keys_true)
    for key in keys_true.keys():
        if key in keys_predict.keys():
            keys_add[key] = keys_predict[key] - keys_true[key]
    # print("!!!!!!!!!!!!!!!!!!!", keys_add.items())
    for add_key, add_val in keys_add.items():
        for true_idx in range(len(temp_y_true) - 1):
            if y_true[true_idx][0] == add_key:
                new_true.append(y_true[true_idx])
                if y_true[true_idx][0] != temp_y_true[true_idx + 1][0]:
                    if add_val < 0:
                        for i in range(abs(add_val)):
                            y_pre.insert(true_idx, [add_key, 0, 0])
                    else: 
                        for i in range(add_val):
                            new_true.append([add_key, 0, 0])
    print(len(new_true), len(y_pre))
    print()
    print(new_true)
    print()
    print(y_pre)
    return new_true

def normalize(value, min_value=0, max_value=300):
    # Ensure value is within the given range
    value = max(min_value, min(value, max_value))
    
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

def get_s4(nrmse, cm):
    tp,fp = cm[0]
    fn,tn = cm[1]

    percision = tp/(tp + fp)
    recall = tp/(tp + fn)

    f1 = (2*percision*recall)/(percision + recall)

    return f1*(1-nrmse)


def get_rmse_confmtrx(prediction_dict, true_keys,
             anomaly_range= CONFIG['anomaly_range'], 
             anomaly_min_time = CONFIG['anomaly_min_time'],
             squared= False):
    y_pre = get_prediction_data(prediction_dict, anomaly_range, anomaly_min_time)
    # y_true = get_true_data(dataset_path)

    matching_true = matching_result(y_pre, true_keys[0], true_keys[1])

    start_true = [[start[0],start[1]] for start in matching_true]
    start_pre = [[start[0],start[1]] for start in y_pre]
    # start_true = [start[2] for start in matching_true]
    # start_pre = [start[2] for start in y_pre]
    rmse = mean_squared_error(start_true, start_pre, squared= squared)

    # rmse = mean_squared_error(matching_true, y_pre, squared= squared)

    true_label = [1 if end[2] > 0 else 0 for end in matching_true]
    pre_label = [1 if end[2] > 0 else 0 for end in y_pre]
    conf_matrix = confusion_matrix(true_label, pre_label, labels= [1, 0])

    nrmse = normalize(rmse)

    s4 = get_s4(nrmse, conf_matrix)
    return s4, nrmse, conf_matrix


def run_weights(iter_range: list, iter_min_time: list, mode, pre_path= PATHS['general'], dataset_path= PATHS['dataset'], select= TRUE_VID):
    rmse_confmtrx_list = []
    
    const_pre = read_prediction_data(pre_path, mode, select)
    
    const_true = get_true_data(dataset_path)
    # print(const_true)
    const_true_keys = kms(const_true)

    for min_time_idx in iter_min_time:
        for range_idx in iter_range:
            rmse_confmtrx = get_rmse_confmtrx(prediction_dict= const_pre, true_keys= (const_true, const_true_keys),
                                    anomaly_range= range_idx,anomaly_min_time= min_time_idx)
            rmse_confmtrx_list.append([rmse_confmtrx[0],[range_idx,min_time_idx], rmse_confmtrx[1], rmse_confmtrx[2]])
    return rmse_confmtrx_list

# const_pre = read_prediction_data("D:\\bi12year3\intern\gpu_slaves\\bau\\", "train", TRUE_VID)
# const_true = get_true_data(PATHS['dataset'])
# const_true_keys = kms(const_true)
# cmtr = get_rmse_confmtrx(prediction_dict= const_pre, true_keys= (const_true, const_true_keys),
#                          anomaly_range= 279,anomaly_min_time= 200)

# print(cmtr)
# cm_display = ConfusionMatrixDisplay(confusion_matrix = cmtr[2])
# cm_display.plot()
# plt.show()

a = run_weights(list(range(1,31)), list(range(1,11)), "train")

print(max(a))


























def get_prediction_data_old(main_path, mode, select= [], 
                        anomaly_range= CONFIG['anomaly_range'], 
                        anomaly_min_time = CONFIG['anomaly_min_time']):
    result_path = f"{main_path}\\results_{mode}"
    predict = []
    # Iterate directory
    for path in os.listdir(result_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(result_path, path)):
            key = int(path.split('_', 4)[2])
            if len(select) != 0 and key not in select:
                pass
            else:
                result_mini = process_ano(key, mode, paths= main_path, 
                                        anomaly_range= anomaly_range,
                                        anomaly_min_time= anomaly_min_time)
                # print(key)
                if len(result_mini.keys()) != 0:
                    for ano_i in result_mini.values():
                        start, end = ano_i[-2][1], ano_i[-1][1]
                        predict.append([key, start, end])
                else:
                    predict.append([key, 0, 0])

    return sorted(predict)

def matching_result_old(y_pre, y_true):
    new_true = []
    keys_true = kms(y_true)
    keys_predict = kms(y_pre)
    keys_add = {}

    temp_y_true = y_true.copy()
    temp_y_true.append([99999, 0, 0])

    for key in keys_true.keys():
        if key in keys_predict.keys():
            keys_add[key] = abs(keys_predict[key] - keys_true[key])

    for add_key, add_val in keys_add.items():
        for true_idx in range(len(temp_y_true) - 1):
            if y_true[true_idx][0] == add_key:
                new_true.append(y_true[true_idx])
                if y_true[true_idx][0] != temp_y_true[true_idx + 1][0]:
                    for i in range(add_val):
                        new_true.append([add_key, 0, 0])
    return new_true


def get_rmse_confmtrx_old(mode, predict_path= PATHS['general'], 
             dataset_path= PATHS['dataset'], 
             anomaly_range= CONFIG['anomaly_range'], 
             anomaly_min_time = CONFIG['anomaly_min_time'],
             select= [], squared= False):
    y_pre = get_prediction_data_old(predict_path, mode, select, anomaly_range, anomaly_min_time)
    y_true = get_true_data(dataset_path)

    matching_true = matching_result_old(y_pre, y_true)

    start_true = [[start[0],start[1]] for start in matching_true]
    start_pre = [[start[0],start[1]] for start in y_pre]
    # start_true = [start[2] for start in matching_true]
    # start_pre = [start[2] for start in y_pre]
    rmse = mean_squared_error(start_true, start_pre, squared= squared)

    # rmse = mean_squared_error(matching_true, y_pre, squared= squared)

    true_label = [0 if end[2] > 0 else 1 for end in matching_true]
    pre_label = [0 for i in range(len(y_pre))]
    conf_matrix = confusion_matrix(true_label, pre_label)

    nrmse = normalize(rmse)

    s4 = get_s4(nrmse, conf_matrix)
    return s4, nrmse, conf_matrix


def run_weights_old(iter_range: list, iter_min_time: list, mode, predicted_path= PATHS['general'], dataset_path= PATHS['dataset']):
    rmse_confmtrx_list = []
     
    for min_time_idx in iter_min_time:
        for range_idx in iter_range:
            rmse_confmtrx = get_rmse_confmtrx_old(mode= mode,
                            predict_path= predicted_path,
                            dataset_path= dataset_path,
                            anomaly_range= range_idx,
                            anomaly_min_time= min_time_idx,
                            select= TRUE_VID)
            
            rmse_confmtrx_list.append([rmse_confmtrx[0],[range_idx,min_time_idx], rmse_confmtrx[1], rmse_confmtrx[2]])
    
    return rmse_confmtrx_list

def get_best_weights_old(max_range, max_time, mode, txt_path= PATHS['general']):
    s4_list = run_weights_old(
            iter_range= max_range,
            iter_min_time= max_time,
            mode= mode,
            predicted_path= txt_path
        )
    return max(s4_list)
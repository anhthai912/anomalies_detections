from mod import process_ano, PATHS, CONFIG
import os
from sklearn.metrics import mean_squared_error, confusion_matrix
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

TRUE_VID = [2, 9, 11, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]

def get_prediction_data(main_path, mode, select= [], 
                        anomaly_range= CONFIG['anomaly_range'], 
                        anomaly_min_time = CONFIG['anomaly_min_time']):
    result_path = f"{main_path}\\results_{mode}"
    predict = []
    # Iterate directory
    for path in os.listdir(result_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(result_path, path)):
            key = int(path.split('_', 4)[2])
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

    predict = sorted(predict)

    if len(select) != 0:
        filter_eval = [sublist for sublist in predict if sublist[0] in select]
        return filter_eval
    else:
        return predict

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
        if lists[i][0] == temp_list[i+1][0]:
            count += 1
            keys[lists[i][0]] = count
        else:
            keys[lists[i][0]] = count
            count = 1
        
    return keys

def matching_result(y_pre, y_true):
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


def get_rmse(mode, predict_path= PATHS['general'], 
             dataset_path= PATHS['dataset'], 
             anomaly_range= CONFIG['anomaly_range'], 
             anomaly_min_time = CONFIG['anomaly_min_time'],
             select= [], squared= False):
    y_pre = get_prediction_data(predict_path, mode, select, anomaly_range, anomaly_min_time)
    y_true = get_true_data(dataset_path)

    matching_true = matching_result(y_pre, y_true)

    start_true = [[start[0],start[1]] for start in matching_true]
    start_pre = [[start[0],start[1]] for start in y_pre]

    # start_true = [start[2] for start in matching_true]
    # start_pre = [start[2] for start in y_pre]
    rmse = mean_squared_error(start_true, start_pre, squared= squared)

    # rmse = mean_squared_error(matching_true, y_pre, squared= squared)

    
    # conf_matrix = confusion_matrix(matching_true, y_pre)
    

    return rmse

# print(get_rmse(mode= "train",
#                 predict_path= "D:\\bi12year3\intern\gpu_slaves\\bau\\",
#                 anomaly_range= 51,
#                 anomaly_min_time= 24,
#                 select= TRUE_VID))



def run_weights(iter_range, iter_min_time, mode, predicted_path= PATHS['general'], dataset_path= PATHS['dataset']):
    rmse_list = []
    for min_time_idx in range(iter_min_time + 1):
        # print(min_time_idx)
        for range_idx in range(iter_range + 5):
            rmse = get_rmse(mode= "train",
                            predict_path= predicted_path,
                            dataset_path= dataset_path,
                            anomaly_range= range_idx,
                            anomaly_min_time= min_time_idx,
                            select= TRUE_VID)
            rmse_list.append([rmse,[range_idx,min_time_idx]])
    
    return rmse_list


def get_best_weights(max_range, max_time, mode, txt_path= PATHS['general']):
    hello = run_weights(
            iter_range= max_range,
            iter_min_time= max_time,
            mode= mode,
            predicted_path= txt_path
        )
    return min(hello)


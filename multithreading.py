# import threading
import multiprocessing
import os
from get_eval import read_prediction_data, get_true_data, kms, get_rmse_confmtrx, PATHS, TRUE_VID, run_weights
from mod import CONFIG
from obj_dect import run_main

def task_post(value_list,function, temp):
    # value = temp[0](temp[1][0], temp[1][1], temp[1][2], temp[1][3])
    range_idx, min_time_idx, mode, path = temp
    value = function(range_idx, min_time_idx, mode, path)
    value_list.extend(value)

def split_list(input_list, n):
    # Calculate the size of each chunk
    avg_size = len(input_list) / n
    # Initialize variables
    out = []
    last = 0.0
    
    while last < len(input_list):
        # Create sublists with lengths rounded to the nearest integer
        out.append(input_list[int(last):int(last + avg_size)])
        last += avg_size
        
    return out

def split_list_batch(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def threader_post(function, iterations:list, iter_index: int, weights, mode, batch_sz:int= 16, pre_path= PATHS['general']):
    manager = multiprocessing.Manager()
    
    # Create a shared list
    value_list = manager.list()

    batch = split_list_batch(iterations, batch_sz)
    threads = []
    print("number of processes: ", len(batch))

    for idx,i in enumerate(batch):
        weights[iter_index] = i
        temp_args = (weights[0], weights[1], mode, pre_path)
        # print(temp_args)
        thread = multiprocessing.Process(target= task_post, args=(value_list, function, temp_args))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return value_list

def task_main(data_list, temp_args, weights):
    mode, conf, show = temp_args
    if weights != None:
        frame_skip, errors, life_time = weights
    else: 
        frame_skip, errors, life_time = CONFIG['frame_skip'], CONFIG['error'], CONFIG['life_time']
    for i in data_list:
        run_main(i, mode, conf, show, frame_skip, errors, life_time)
    

def threader_main(data_list:list, mode, weights= None, batch_sz:int= 4):
    manager = multiprocessing.Manager()
    
    # Create a shared list
    value_list = manager.list()

    batch = split_list(data_list, batch_sz)
    threads = []
    print("number of processes: ", len(batch))

    
    for idx,i in enumerate(batch):
        temp_args = (mode, 0.5, False)
        # print(temp_args)
        thread = multiprocessing.Process(target= task_main, args=(i, temp_args, weights))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return value_list

if __name__ == '__main__':
    iter_range1 = list(range(1,31))
    iter_min_time1 = list(range(1,18))
    mode1 = "train"
    # pre_path1 = "D:\\bi12year3\intern\gpu_slaves\\bau\\"
    full_data = list(range(1,101))


    super_args = [iter_range1, iter_min_time1]

    # threader_main(TRUE_VID, "train", batch_sz= 4)
    print("***************************************")

    result = threader_post(run_weights, iter_min_time1, 1, super_args, "train")

    print(max(result))












# Define the function to be executed by each process
# def process_function(process_id):
#     print(f"Process {process_id} is running")

# # Number of processes
# def lmeo():
#     n = 5

#     # Create a list to hold the process objects
#     processes = []

#     # Create and start n processes
#     for i in range(n):
#         process = multiprocessing.Process(target=process_function, args=(i,))
#         processes.append(process)
#         process.start()

#     # Optionally, wait for all processes to finish
#     for process in processes:
#         process.join()

#     print("All processes have finished execution")

# def run_weights_threads(iter_range: list, iter_min_time: list, mode, batch_sz:int= 16, pre_path= PATHS['general'], dataset_path= PATHS['dataset'], select= TRUE_VID):
    
    
#     const_pre = read_prediction_data(pre_path, mode, select)
#     const_true = get_true_data(dataset_path)
#     const_true_keys = kms(const_true)

#     rmse_confmtrx_list = threader(run_weight_mini, iter_min_time,
#                                   (iter_range, const_pre, const_true, const_true_keys),
#                                   batch_sz)

#     # for range_idx in iter_range:
#     #     for min_time_idx in iter_min_time:
#     #         rmse_confmtrx = get_rmse_confmtrx(prediction_dict= const_pre, true_keys= (const_true, const_true_keys),
#     #                                 anomaly_range= range_idx,anomaly_min_time= min_time_idx)
#     #         rmse_confmtrx_list.append([rmse_confmtrx[0],[range_idx,min_time_idx], rmse_confmtrx[1], rmse_confmtrx[2]])
#     return rmse_confmtrx_list

# def run_weight_mini(iter_range, const_pre, const_true, const_true_keys, min_time_idx):
#     score = []
#     for range_idx in iter_range:
#         rmse_confmtrx_mini = get_rmse_confmtrx(prediction_dict= const_pre, true_keys= (const_true, const_true_keys),
#                                 anomaly_range= range_idx,anomaly_min_time= min_time_idx)
#         score.append([rmse_confmtrx_mini[0],[range_idx,min_time_idx], rmse_confmtrx_mini[1], rmse_confmtrx_mini[2]])
#     return score














# lmoa = threader(task1, list(range(1,11)), [[1,2,3], [7,9], [0,89,6,9]],5)

# print(lmoa)
# if __name__ == "__main__":

#     print("ID of process running main program: {}".format(os.getpid()))
#     print("Main thread name: {}".format(threading.current_thread().name))

#     list1 = [1,2,3,4,5,6,7,8,9,10]
#     batch_sz = 4

#     new_list = []

#     threads = []

#     batch = split_list(list1, batch_sz)
    
#     for idx, i in enumerate(batch):
#         print(i)
#         # process = multiprocessing.Process(target=task1, args=(i,))
#         process = threading.Thread(target=task1, name= idx+1, args=(i,))
#         threads.append(process)
#         process.start()

#     # Optionally, wait for all threads to finish
#     for thread in threads:
#         thread.join()
        

#     print(sorted(new_list))

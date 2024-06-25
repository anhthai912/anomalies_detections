# import threading
import multiprocessing
import os
# from get_eval import read_prediction_data, get_true_data, kms, get_rmse_confmtrx, PATHS, TRUE_VID, run_weights

def task(value_list,temp):
    value = temp[0](temp[1][0], temp[1][1], temp[1][2], temp[1][3])
    value_list.extend(value)

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def threader(function, iterations:list, args, batch_sz:int= 16):
    manager = multiprocessing.Manager()
    
    # Create a shared list
    value_list = manager.list()

    batch = split_list(iterations, batch_sz)
    threads = []
    print("number of processes: ", len(batch))
    
    for idx,i in enumerate(batch):
        # print(i)
        temp_args = (args[0], i, args[2], args[3])
        # print(temp_args)
        thread = multiprocessing.Process(target= task, args=(value_list, (function, temp_args)))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return value_list



if __name__ == '__main__':
    iter_range1 = list(range(1,31))
    iter_min_time1 = list(range(1,11))
    mode1 = "train"
    pre_path1 = "D:\\bi12year3\intern\gpu_slaves\\bau\\"

    super_args = (iter_range1, iter_min_time1, mode1, pre_path1)

    result = threader(run_weights, iter_min_time1, super_args, 4)

    print(max(result))







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

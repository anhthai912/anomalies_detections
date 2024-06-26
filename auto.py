from obj_dect import run_main
from mod import PATHS, CONFIG
from get_eval import run_weights
from multithreading import threader_main, threader_post
import time

data = [2, 9, 11, 14, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
iter_range = list(range(0,101, 5))
error = 1
iter_min_time = list(range(1,401, 10))
life_time = list(range(1, 91, 15))
frame_skip = [1, 2, 3, 4]
mode = "train"
weights = [iter_range, iter_min_time]
# data = [93,95, 97]
# data = [2, 9, 11, 14, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
# for i in data:
#     print(i)
#     run_main(33, "train", show= True, confident= 0.5)
if __name__ == '__main__':
    # for i in data:
    #     print(i)
    
    # ouput_path = PATHS["general"] + f'results\\{mode}_Output_anomalies.txt'
    iteration = 0
    training_data = []

    # fr_skp = 1 #khang
    # fr_skp = 2 #hminh
    fr_skp = 3 #me
    # fr_skp = 4 #bau
    beginer = time.time()
    for life in life_time:
        start = time.time()
        weights_idx = [fr_skp, error, life]
        threader_main(data, mode, weights_idx, 4)
        result_list = threader_post(run_weights, iter_min_time, 1, weights, mode, 10)

        result = max(result_list)
        print(result)

        s4, [ano_range, min_time], rmse, cm = result
        w_data = [s4, [error, life, fr_skp, ano_range, min_time], rmse, cm]
        training_data.append(w_data)

        end = time.time()
        timer = end - start
        timer = round(timer, 2) 

        iteration += 1
        ouput_path_i = PATHS["general"] + f'results\\{mode}_Output_{iteration}_weights.txt'
        text_file = open(ouput_path_i, 'w')
        for i in w_data:
            text_file.write(f"{str(i)}\n")
        text_file.write(f"\nStart to end:{timer}")
        text_file.closed


    print("DONEZO@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    if len(training_data) == 0:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    else: 
        ended = time.time()
        full_time = ended - beginer
        full_time = round(timer, 2)
        # timer = end - start
        # timer = round(timer, 2) 
        ouput_path_full = PATHS["general"] + f'results\\{mode}_Output_best_weights.txt'
        text_file_full = open(ouput_path_full, 'w')
        for i in training_data:
            text_file_full.write(f"{str(i)}\n")
        text_file_full.write(f"\nStart to end:{full_time}")
        text_file_full.write(f"\nBest score:\n {max(training_data)}\n")
        text_file_full.write(f"\nBest weights: {max(training_data)[1]}\n")
        text_file_full.write(f"\nnumber of iterations {str(len(training_data))}")

        text_file_full.close() 
        print("Best weight:", max(training_data))


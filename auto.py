from obj_dect import run_main
from mod import PATHS, CONFIG
from get_eval import run_weights
from multithreading import threader_main, threader_post


data = [2, 9, 11, 14, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
iter_range = list(range(1,51))
iter_min_time = list(range(1,201))
errors = [0, 1, 2]
life_time = list(range(15, 91, 15))
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
    ouput_path = PATHS["general"] + f'results\\{mode}_Output_anomalies.txt'
    text_file = open(ouput_path, 'w')
    training_data = []

    # fr_skp = 1 #khang
    # fr_skp = 2 #hminh
    fr_skp = 3 #me
    # fr_skp = 4 #bau
    for error in errors:
        for life in life_time:
            weights_idx = [fr_skp, error, life]
            threader_main(data, mode, weights_idx, 4)
            result_list = threader_post(run_weights, iter_min_time, 1, weights, mode, 50)

            result = max(result_list)
            s4, [ano_range, min_time], rmse, cm = result

            w_data = [s4, [error, life, fr_skp, ano_range, min_time], rmse, cm]

            training_data.append(w_data)

    
    for i in training_data:
        text_file.write(f"{str(i)}\n")
    text_file.write(f"\nBest score:\n {max(training_data)}\n")
    text_file.write(f"\nnumber of anomalies {str(len(training_data))}")

    text_file.close() 

    print("DONEZO@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Best weight:", max(training_data))


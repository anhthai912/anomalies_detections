from obj_dect import run_main
from mod import PATHS, CONFIG
from get_eval import run_weights
from multithreading import threader


data = [2, 9, 11, 14, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
# data = [93,95, 97]
# data = [2, 9, 11, 14, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97]
# for i in data:
#     print(i)
#     run_main(33, "train", show= True, confident= 0.5)
if __name__ == '__main__':
    iter_range = list(range(1,51))
    iter_min_time = list(range(1,201))
    errors = [0, 1 , 2]
    life_time = list(range(15, 90, 15))
    frame_skip = [0, 1, 2, 3, 4, 5]
    mode = "train"
    weights = [iter_range, iter_min_time]

    for i in data:
        print(i)
        run_main(i, "train", show= False, confident= 0.5)

    result = threader(run_weights, iter_min_time, 1, weights, "train", 32, pre_path= "D:\\bi12year3\intern\gpu_slaves\\bau\\")

    print(max(result))
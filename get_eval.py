from mod import process_ano
import os
from sklearn.metrics import mean_squared_error

mode = "train"

result_path = r'D:\\bi12year3\\intern\\ictlab\\imgonnacrylmao\\results_train'

predict = []

# Iterate directory
for path in os.listdir(result_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(result_path, path)):
        key = int(path.split('_', 4)[2])
        result_mini = process_ano(key, mode)
        # print(key)
        if len(result_mini.keys()) != 0:
            for ano_i in result_mini.values():
                start, end = ano_i[-2][1], ano_i[-1][1]
                predict.append([key, start, end])

for i in predict:
    print(i)
        

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


eval_path = "C:\\Users\ADMIN\Desktop\AIC21-Track4-Anomaly-Detection_full\AIC21-Track4-Anomaly-Detection\\train-anomaly-results.txt"

eval = []

eval_file = open(eval_path, 'r')
for i in eval_file:
    vid, start, end = i.split(' ', 2)
    end = end.strip('\n')
    vid,start, end = int(vid),int(start), int(end)
    eval.append([vid,start,end])
eval_file.close()


filter_eval = [sublist for sublist in eval if sublist[0] == 33 or sublist[0] == 2]
for i in filter_eval:
    print(i)

def kms(lists):
    keys = {}
    count = 1
    for i in range(len(lists) - 1):
        if lists[i][0] == lists[i+1][0]:
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

    # for true_idx in range(len(y_true)):
    #     if y_true[true_idx][0]
    for add_key, add_val in keys_add.items():
        for true_idx in range(len(temp_y_true) - 1):
            if y_true[true_idx][0] == add_key:
                new_true.append(y_true[true_idx])
                if y_true[true_idx][0] != temp_y_true[true_idx + 1][0]:
                    for i in range(add_val):
                        new_true.append([add_key, 0, 0])

    
    return new_true

done = matching_result(predict, filter_eval)
# done = kms(predict)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(len(predict))
# print(kms(predict))
# print(kms(filter_eval))
for i in done:
    print("this is ", i)
print(len(done) - len(predict))

rmse = mean_squared_error(done, predict, squared= False)
# print(done)
print(rmse)
        
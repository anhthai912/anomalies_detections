# Extract valid predictions
predictions = [
    [2, 500, 0], [9, 0, 0], [11, 0, 0], [14, 0, 0], [33, 2, 12], [33, 160, 894], [35, 0, 0], [49, 664, 894], [49, 805, 894],
    [51, 0, 0], [63, 0, 0], [72, 106, 894], [73, 0, 0], [74, 294, 894], [83, 0, 0], [91, 635, 900], [93, 4, 796],
    [93, 812, 892], [95, 89, 894], [95, 172, 894], [95, 178, 894], [95, 182, 291], [95, 242, 335], [95, 246, 258],
    [95, 280, 295], [97, 18, 833]
]
true_anomalies = [
    [2, 587, 894], [9, 0, 287], [11, 0, 888], [14, 475, 600], [33, 0, 90], [33, 165, 894], [35, 106, 185], [49, 422, 894],
    [51, 431, 891], [63, 87, 853], [72, 87, 894], [73, 155, 894], [74, 293, 894], [83, 540, 892], [91, 602, 900],
    [93, 0, 892], [95, 38, 894], [97, 0, 890]
]
t = True
f = False
check= {
    2: t,
    9: t,
    11: t,
    14: f,
    33: t, 
    33: t,
    35: f,


}
# predictions = [
#     [2, 587, 894], [9, 0, 287], [11, 0, 888], [14, 475, 600], [33, 0, 90], [33, 165, 894], [35, 106, 185], [49, 422, 894],
#     [51, 431, 891], [63, 87, 853], [72, 87, 894], [73, 155, 894], [74, 293, 894], [83, 540, 892], [91, 602, 900],
#     [93, 0, 892], [95, 38, 894], [97, 0, 890]
# ]


valid_predictions = [pred for pred in predictions if pred[1] != 0 or pred[2] != 0]
valid_true_anomalies = [true for true in true_anomalies]

# Group predictions and true anomalies by video key
from collections import defaultdict

pred_dict = defaultdict(list)
true_dict = defaultdict(list)

for pred in valid_predictions:
    pred_dict[pred[0]].append(pred)

for true in valid_true_anomalies:
    true_dict[true[0]].append(true)

TP = 0
FP = 0
FN = 0
# matched_true_indices = set()
detection_times = []

for key in pred_dict:
    matched_true_indices = set()
    if key in true_dict:
        # print("ehll",key)
        for pred in pred_dict[key]:
            best_match = None
            min_time_diff = float('inf')
            for i, true in enumerate(true_dict[key]):
                # print(true)
                # print(matched_true_indices)
                if i not in matched_true_indices:
                    time_diff = abs(pred[1] - true[1])
                    # print(time_diff)
                    if time_diff <= 100 and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = i
            if best_match is not None:
                TP += 1
                matched_true_indices.add(best_match)
                detection_times.append(min_time_diff)
            else:
                FP += 1
        FN += len(true_dict[key]) - len(matched_true_indices)
    else:
        FP += len(pred_dict[key])

for key in true_dict:
    if key not in pred_dict:
        FN += len(true_dict[key])

# Compute Precision, Recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


rmse = np.sqrt(np.mean(np.square(detection_times))) if detection_times else 0
rmse = normalize(rmse)

print(f1_score*(1-rmse))
# print(float('inf'))
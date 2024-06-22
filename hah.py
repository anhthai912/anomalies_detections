# import random
from collections import OrderedDict
# import supervision as sv
# import numpy as np
# track_id = OrderedDict([
#     ("1", [2, 4, 0, 0]),
#     ("2", [4, 5, 20, 0]),
#     ("5", [10, 8, 6, 0])
# ])
# # track_id = OrderedDict()

# anomalies = {}
# def coor_check(x_old, y_old, x_nw, y_nw):
#     if x_old == x_nw and y_old == y_nw:
#         return True
#     else:
#         return False

# def add_anomaly(vehicle, anomalies: dict, track_ids):
#     ano = anomalies.copy()
#     track = track_ids.copy()
#     if vehicle[1][3] == 1:
#         ano[vehicle[0]] = vehicle[1]
#         # track.pop(vehicle[0])
#     return ano, track

# def check_life(tracker_id:dict, anomalies: dict):
#     for life in tracker_id.items():
#         if int(life[1][2]) > 10:
#             life[1][3] = 1
#             anomalies, tracker_id = add_anomaly(life, anomalies, tracker_id)
#     return anomalies, tracker_id

# def update_checker(tracker_id: dict, Id, x_nw, y_nw):
#     x_old, y_old, life, anomaly = tracker_id[Id]
#     if coor_check(x_old, y_old, x_nw, y_nw):
#         life += 1
#         # track_id[Id] = [x_nw, y_nw, life]
#         tracker_id.update({Id: [x_nw, y_nw, life, anomaly]})
#         return tracker_id
#     else:
#         life = 0
#         tracker_id[Id] = [x_nw, y_nw, life, anomaly]
#         return tracker_id
        
# c = 0
# print(track_id)
# for i in range(30):
#     # c += 1
#     # x, y = random.randint(2,50), random.randint(2,50)
#     # if str(i) not in track_id:
#     #     track_id[str(i)]= [x, y, 0]
#     #     if len(track_id.items()) >= 10:
#     #         anomalies.append(track_id.popitem(last = False))
          
#     # else:
#     #     track_id = update_checker(track_id, "1", 2,4)
    
#     # check_life(track_id)
#     #     print(track_id, "lmaoooo")
#     #     break
# # track_id.popitem(last= False)   

#     tracker_id = "8"
#     x = 6
#     y = 5
#     if tracker_id not in track_id:
#         track_id[tracker_id] = [x, y, 0, 0]
#         if len(track_id.items()) > 2:
#             a = track_id.popitem(last= False)
#             if a[0] in anomalies.keys():
#                 track_id[a[0]] = a[1]
#     else:
#         track_id = update_checker(track_id, tracker_id, x, y)

#     anomalies, track_id = check_life(track_id, anomalies)
        

# print(track_id)
# print("hererpppppppppppppppppppppppppppppppp", anomalies)
    
# print(sv.ColorPalette.from_matplotlib('viridis', 1000))

# hehe = OrderedDict([('0', [30, 39, 0]), ('1', [19, 7, 0]), ('2', [33, 46, 0]), ('3', [4, 31, 0]), ('4', [44, 41, 0]), ('5', [28, 30, 0]), ('6', [33, 18, 0]), ('7', [29, 32, 0]), ('8', [7, 38, 0]), ('9', [46, 40, 0]), ('10', [3, 47, 0]), ('11', [50, 6, 0]), ('12', [3, 8, 0]), ('13', [50, 36, 0]), ('14', [45, 31, 0]), ('15', [45, 27, 0]), ('16', [24, 45, 0]), ('17', [40, 31, 0]), ('18', [48, 10, 0]), ('19', [28, 7, 0])])


lmao = OrderedDict([(78907, [290, 345, 355, 1]),
(77722, [270, 364, 377, 1]),
(79118, [290, 345, 414, 0]),
(79609, [291, 346, 444, 0])])

for life in lmao.items():
    print(life)
    if int(life[1][3]) == 0:
        print((lmao[life[0]][:-1]), "asdffafaffafd")
        lmao[life[0]][2] = 1

print(lmao)

# print(len(hehe.items()))
# print(hehe.popitem(last= False))
# np.asarray(lmao, )


{9: [[317, 113], 294, [1, 0], [1811, 60]], 
 10: [622, 53, 107, [215, 7], [329, 11]], 
 32: [621, 48, 123, [573, 19], [708, 24]], 
 124: [526, 22, 137, [581, 19], [718, 24]], 
 147: [413, 55, 268, [880, 29], [1148, 38]], 
 171: [591, 55, 169, [1425, 48], [1884, 63]], 
 182: [536, 69, 152, [1437, 48], [1592, 53]], 
 174: [593, 42, 162, [1489, 50], [1884, 63]], 
 226: [570, 57, 122, [1619, 54], [1741, 58]]}


from math import isclose
list1 = [1,2,3,5]
list2 = [1,2,3,5]
check = 0
for i in range(len(list1)):
    if list1[i] == list2[i]:
        check += 1
print(check == 4)

hihi = [[317, 113], 294, [1, 0], [1811, 60]]

print(hihi[:2])
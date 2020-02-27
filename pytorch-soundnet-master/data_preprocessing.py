import cv2
import numpy as np
import json

import re

label = ['0', '1', '2']
contact1 = [249, 291, 328, 368, 406, 444, 482, 520, 559, 942, 1203, 1240, 1503, 1614, 1727, 1802, 1876, 1949, 2060,
           2207, 2353, 2424, 2530, 2602, 2710, 2783, 2925, 2296, 3066, 3138, 3279, 3388, 3493, 3636, 3709, 3781, 3853, 3924,
           3996, 4136, 4245, 4351, 4422, 4490, 4528, 4633, 4667, 5541, 5609, 5679, 5752, 5822, 5893, 5995, 6064, 6166, 6235, 6306, 6415, 6520]
# contact =  [250,291,329,368,406,443,481,520,558,596,633,671,710,749,788,827,865,903,941,978,1015,1052,1089,1127,1164,1202,1239,
#             1276,1313,1350,1388,1426,1464,1501,1539,1576,1614,1651,1689,1725,1763,1800,1837,1875,1911,1948,1985,2022,2059,2096,
#             2132,2169,2206,2242,2279,2315,2352,2388,2423,2459,2494,2530,2566,2602,2637,2674,2710,2745,2782,2818,2854,2890,2925,
#             2960,2996,3031,3066,3102,3138,3173,3209,3244,3279,3315,3351,3386,3421,3457,3493,3529,3565,3601,3636,3672,3709,3745,
#             3781,3817,3853,3889,3924,3960,3996,4031,4066,4101,4136,4171,4206,4242,4278,4312,4349,4384,4420,4455,4490,4525,4560,
#             4596,4631,4667,4701,4737,4772,4808,4843,4879,4915,4950,4985,5021,5057,5093,5128,5163,5199,5234,5269,5303,5338,5371,
#             5405,5438,5473,5507,5541,5575,5609,5644,5679,5715,5751,5786,5821,5856,5891,5927,5961,5995,6030,6064,6098,6132,6166,
#             6200,6235,6271,6306,6342,6378,6413,6449,6485,6520,6555,6590,6625,6660,6695,6730,6766,6803,6841,6879,6918,6968,7020,
#             7080,7136,7450,7504,7543,7582,7620,7657,7694,7733,7771,7809,7847,7885,7923,7962,8000,8040,8079,8116,8155,8192,8229,
#             8266,8303,8340,8378,8415,8453,8490,8527,8564,8601,8640,8677,8714,8752,8790,8828,8865,8902,8939,8977,9014,9051,9088,
#             9125,9162,9199,9236,9273,9310,9346,9383,9420,9456,9493,9529,9566,9602,9638,9674,9709,9744,9780,9816,9852,9887,9924,
#             9960,9996,10032,10068,10104,10140,10175,10210,10245,10280,10316,10352,10387,10423,10459,10494,10529,10565,10600,
#             10636,10671,10707,10743,10779,10815,10850,10887,10923,10959,10995,11031,11068,11103,11139,11175,11210,11246,11281,
#             11316,11351,11386,11420,11456,11492,11527,11563,11599,11634,11669,11704,11740,11775,11810,11846,11881,11916,11951,
#             11987,12022,12057,12093,12129,12165,12200,12235,12271,12307,12343,12377,12413,12448,12483,12518,12553,12587,12620,
#             12654,12688,12722,12756,12790,12824,12858,12894,12929,12965,13000,13035,13070,13106,13141,13176,13210,13244,13279,13313,
#             13347,13381,13415,13450,13484,13520,13555,13591,13628,13663,13699,13734,13770,13805,13840,13875,13910,13945,13980,14017,
#             14054,14092,14130,14175,14225,14278,14340,14396]

def load_imu_TrainSeq(path, train):
    if train:
        datafile = 'train_max.txt'
        labelfile = 'trainlabel_max.txt'
        output = '\\imu_train_data.json'
    else:
        datafile = 'validate.txt'
        labelfile = 'validatelabel.txt'
        output = '\\imu_validate_data.json'

    Sequence = []
    contact = []
    teo = []
    count  = 0
    label = 0
    labelf = open(path + '\\' + labelfile)
    line = labelf.readline()   #'22, 1\n'
    while line:
        line = labelf.readline()
        seq = line.split('\n')
        seq = seq[0].split(',')
        if len(seq) <= 1:
            continue
        if int(seq[1]) == 1:
            contact.append(int(seq[0]))
        if int(seq[1]) == 2:
            teo.append(int(seq[0]))
    labelf.close()


    dataf = open(path + '\\' + datafile)
    line = dataf.readline()
    seg_index = []
    while line:
        count = count + 1
        print(count)
        if count-3 in contact:
            label = 1
            seg_index.append(count-3)
        elif count in teo:
            label = 2
        else:
            label = 0
        line = dataf.readline()
        seq = line.split(",")
        if len(seq) <= 1:
            continue
        dot = {
            "Num": count-1,
            "NodeID": seq[0],
            "systic": seq[1],
            "timestamp": seq[2],
            "accx": seq[3],
            "accy": seq[4],
            "accz": seq[5],
            "gyrox": seq[6],
            "gyroy": seq[7],
            "gyroz": seq[8],
            "qw": seq[9],
            "qx": seq[10],
            "qy": seq[11],
            "qz": seq[12],
            "angle": seq[13],
            "label": label,
        }
        Sequence.append(dot)
    dataf.close()

    Seg_sequence = [Sequence[(seg_index[i]-8):(seg_index[i+1])] for i in range(len(seg_index)-1) if (seg_index[i+1]-seg_index[i])>30 & (seg_index[i + 1] - seg_index[i]) < 72]

    with open(('C:\\ALEX\\Doc\\Reference\\SoundNet\\Dataset'+output), "w") as jf:
        json.dump(Seg_sequence, jf)

        print("加载入文件完成...")


def load_imu_validateSeq(path):

    validate = 'validate.txt'
    validatelabel = 'validatelabel.txt'

    Sequence = []
    contact = []
    teo = []
    count = 0
    label = 0
    validatelabelf = open(path + '\\' + validatelabel)
    line = validatelabelf.readline()
    while line:
        line = validatelabelf.readline()
        seq = line.split('\n')
        seq = seq[0].split(',')
        if len(seq) <= 1:
            continue
        if int(seq[1]) == 1:
            contact.append(int(seq[0]))
        if int(seq[1]) == 2:
            teo.append(int(seq[0]))
    validatelabelf.close()


    validatef = open(path + '\\' + validate)
    line = validatef.readline()
    seg_index = []
    while line:
        count = count + 1
        print(count)
        if count - 3 in contact:
            label = 1
            seg_index.append(count - 3)
        elif count in teo:
            label = 2
        else:
            label = 0
        line = validatef.readline()
        seq = line.split(",")
        if len(seq) <= 1:
            continue
        dot = {
            "Num": count - 1,
            "NodeID": seq[0],
            "systic": seq[1],
            "timestamp": seq[2],
            "accx": seq[3],
            "accy": seq[4],
            "accz": seq[5],
            "gyrox": seq[6],
            "gyroy": seq[7],
            "gyroz": seq[8],
            "qw": seq[9],
            "qx": seq[10],
            "qy": seq[11],
            "qz": seq[12],
            "angle": seq[13],
            "label": label,
        }
        Sequence.append(dot)
    validatef.close()

    Seg_sequence = [Sequence[(seg_index[i] - 8):(seg_index[i + 1] )] for i in range(len(seg_index) - 1) if
                    (seg_index[i + 1] - seg_index[i]) > 30 & (seg_index[i + 1] - seg_index[i]) < 72]
    with open(('C:\\ALEX\\Doc\\Reference\\SoundNet\\Dataset'+'\\imu_validate_data.json'), "w") as jf:
        json.dump(Seg_sequence, jf)

        print("加载入文件完成...")

path = "C:\\ALEX\\Doc\\Reference\\SoundNet\\Dataset"
# load_imu_seq(path)
# load_imu_TrainSeq(path, True)
load_imu_TrainSeq(path, True)

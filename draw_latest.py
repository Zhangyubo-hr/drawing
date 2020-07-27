import json
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import deque


def read_json(path):
    dict = json.load(open(path))
    return dict

def np_move_avg(a, n, mode = "same"):
    return (np.convolve(a, np.ones((n,)) / n, mode = mode))

def calculate_dis(center1,center2):
    center1 = np.array(center1)
    center2 = np.array(center2)
    return np.sqrt(np.sum(np.square(center1 - center2)))


# ball json -> dict
ballDict = read_json("./json/ball_personid.json")
personDict = read_json("./json/poseAdd2Id.json")

# color for different person
# COLORS = [[0, 255, 0], [0, 0, 255], [255, 0, 255], [255, 255, 0], [0, 255, 255]]
COLORS = [[120, 220, 0], [100, 100, 255], [150, 80, 80], [255, 200, 100], [0, 180, 255]]
# COLORS = [[100, 200, 100], [100, 100, 200], [200, 100, 200], [200, 200, 100], [100, 200, 200]]

# cv2 input, output, para ...
capture = cv2.VideoCapture("../videos/ori.avi")
w = int(capture.get(3))
h = int(capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("../test27.mp4", fourcc, 30, (w, h))

# ball track queue
xballQue = [deque(maxlen = 10) for _ in range(10)]
yballQue = [deque(maxlen = 10) for _ in range(10)]
xpersonQue = [deque(maxlen = 10) for _ in range(10)]
ypersonQue = [deque(maxlen = 10) for _ in range(10)]

# in order to draw ball track
pts = [deque(maxlen=30) for _ in range(99)]
time_since_update = {}
for _ in range(60):
    time_since_update[str(_)] = 100


# smooth id 
# smooth id 
posX = np.zeros((1097, 6))
posY = np.zeros((1097, 6))
smoothX = [[], [], [], [], [], []]
smoothY = [[], [], [], [], [], []]

smooT =[0, 0, 0, 0, 0]
# 0 - 1096
for kFrame, vFrame in personDict.items():
    # 1 - 5
    for kPerson, vPerson in vFrame.items():
        xx = (vPerson[0] + vPerson[2]) / 2
        yy = (vPerson[1] + vPerson[3]) / 2
        dx = (vPerson[2] - vPerson[0]) / 2
        dy = (vPerson[1] - vPerson[3]) / 2
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if kFrame == '0':
            smooT[int(kPerson) - 1] = dist
        smooT[int(kPerson) - 1] = (smooT[int(kPerson) - 1] + dist) / 2
        x = xx
        y = yy - smooT[int(kPerson) - 1]
        posX[int(kFrame)][int(kPerson)] = 1
        smoothX[int(kPerson)].append(x)
        posY[int(kFrame)][int(kPerson)] = 1
        smoothY[int(kPerson)].append(y)

smoothSize = 20
for i in [1, 2, 3, 4, 5]:
    smoothX[i] = np_move_avg(smoothX[i], smoothSize, "same")
    smoothY[i] = np_move_avg(smoothY[i], smoothSize, "same")

for i in [1, 2, 3, 4, 5]:
    count = 0
    for j in range(1097):
        if posX[j][i] == 1:
            posX[j][i] = smoothX[i][count]
            posY[j][i] = smoothY[i][count]
            count = count + 1





# 逐帧遍历 videos, write some json in it
# 0 - 1096 
frame_num = 0
smooT2 = [0, 0 ,0, 0, 0]
while True:
    #print("frame: {%d}"%frame_num)
    ret, frame = capture.read()
    if ret != True:
        break
    oriFrame = frame.copy()
    
    # read json write in frame
    ballInfo = ballDict[str(frame_num)]
    personInfo = personDict[str(frame_num)]

    # 画球
    for kBall, vBall in ballInfo.items():
        if len(vBall) == 0:
            continue
        track_id = int(kBall)
        idBall = int(int(kBall) / 10)
        color = COLORS[idBall - 1]
        # 小球的框
        # cv2.rectangle(frame, (int(vBall[0]), int(vBall[1])), (vBall[2], vBall[3]), color, 3)
        # 小球的轨迹
        center = (int((vBall[0] + vBall[2]) / 2), int((vBall[1] + vBall[3]) / 2))
        time_since_update[str(track_id)] = 0
        pts[track_id].append(center)

    for i in range(60):
        time_since_update[str(i)] += 1
        if(time_since_update[str(i)] >= 3):
            for _ in range(3):
                if len(pts[i]) == 0:
                    break
                pts[i].popleft()
    
    for _id in range(60):
        colorB = COLORS[int(_id / 10) - 1].copy()
        if len(pts[_id]) == 0:
            continue
        for j in range(len(pts[_id]) - 1, -1, -1):
            #if pts[_id][j] is None:
            #    continue
            thickness = int(np.sqrt(20 + j * 1))
            #cv2.circle(frame, (pts[track_id][j]), 2, (color), thickness)
            for _c in range(3):
                if colorB[_c] < 200:
                    colorB[_c] = colorB[_c] + 2
                else:
                    colorB[_c] = colorB[_c] - 2
            cv2.circle(frame, (pts[_id][j]), thickness, (colorB),  -1)


    # 画人的ID()
    for kPerson, vPerson in personInfo.items():
        xx = int((vPerson[0] + vPerson[2]) / 2)
        yy = int((vPerson[1] + vPerson[3]) / 2)
        dx = (vPerson[2] - vPerson[0]) / 2
        dy = (vPerson[3] - vPerson[1]) / 2
        dist = np.sqrt(dx ** 2 +  dy ** 2)
        # print(dist)
        # 平滑半径 
        if kFrame == '0':
            smooT2[int(kPerson) - 1] = dist
        smooT2[int(kPerson) - 1] = (smooT2[int(kPerson) - 1] + dist) / 2
        
        # 透明操作
        colorG = COLORS[int(kPerson) - 1]
        # cv2.circle(oriFrame, (xx, yy), int(dist), (colorG), 10)
        # cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)
        if frame_num > 50 and frame_num < 75 and kPerson == '3':
            cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
            cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        # elif frame_num > 180 and frame_num < 260 and kPerson == '1':
        #     cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
        #     cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        elif frame_num > 280 and frame_num < 350 and kPerson == '5':
            cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
            cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        # elif frame_num > 480 and frame_num < 570 and kPerson == '4':
        #     cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
        #     cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        # elif frame_num > 680 and frame_num < 760 and kPerson == '5':
        #     cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
        #     cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        elif frame_num > 1045 and kPerson == '2':
            cv2.circle(oriFrame, (xx, yy), int(dist), colorG, 10)  # 圆框
            cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)  # 圆内部
        # cv2.circle(oriFrame, (xx, yy), int(dist), (colorG), 10)
        # cv2.circle(oriFrame, (xx, yy), int(dist), (255, 255, 255), -1)
         

        #平滑处理
        xTemp = xx
        yTemp = yy - smooT2[int(kPerson) - 1]
        xpersonQue[int(kPerson)].append(xTemp)
        ypersonQue[int(kPerson)].append(yTemp)
        xSum = 0
        ySum = 0
        for _it in range(len(xpersonQue[int(kPerson)])):
            xSum = xSum + xpersonQue[int(kPerson)][_it]
            ySum = ySum + ypersonQue[int(kPerson)][_it]
        xText1 = xSum / len(xpersonQue[int(kPerson)])
        yText1 = ySum / len(xpersonQue[int(kPerson)])

        if frame_num < 1090:
            xText2 = posX[frame_num][int(kPerson)]
            yText2 = posY[frame_num][int(kPerson)]
        else:
            xText2 = xTemp
            yText2 = yTemp
        xText = (xText1 + xText2) / 2
        yText = (yText1 + yText2) / 2
        colorT = COLORS[int(kPerson) - 1]
        cv2.putText(frame, kPerson, (int(xText), int(yText)), 0, 5e-3 * 150, (255, 255, 255), 6)
        #cv2.putText(frame, kPerson, (int(xText), int(yText)), 0, 5e-3 * 150, (255, 100, 0), 3)
        cv2.putText(frame, kPerson, (int(xText), int(yText)), 0, 5e-3 * 150, (colorT), 3)

    img_mix = cv2.addWeighted(frame, 0.8, oriFrame, 0.4, 0)
    cv2.imshow("test", img_mix)
    #cv2.imshow("test", frame)
    #cv2.imshow("test", oriFrame)
    #out.write(frame) 
    out.write(img_mix) 
    frame_num = frame_num + 1
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
out.release()

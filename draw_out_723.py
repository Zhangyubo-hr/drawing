import json
import cv2
import os
import numpy as np
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
COLORS = [[0, 255, 0], [0, 0, 255], [255, 0, 255], [255, 255, 0], [0, 255, 255]]

# cv2 input, output, para ...
capture = cv2.VideoCapture("./ori.avi")
w = int(capture.get(3))
h = int(capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("test21.mp4", fourcc, 30, (w, h))

# ball track queue
xballQue = [deque(maxlen = 10) for _ in range(10)]
yballQue = [deque(maxlen = 10) for _ in range(10)]
xpersonQue = [deque(maxlen = 10) for _ in range(10)]
ypersonQue = [deque(maxlen = 10) for _ in range(10)]

# in order to draw ball track
pts = [deque(maxlen=10) for _ in range(9999)]
time_since_update = {}
for _ in range(60):
    time_since_update[str(_)] = 100


# 逐帧遍历 videos, write some json in it
# 0 - 1096 
frame_num = 0
while True:
    print("frame: {%d}"%frame_num)
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
        #cv2.rectangle(frame, (int(vBall[0]), int(vBall[1])), (vBall[2], vBall[3]), color, 3)
        # 小球的轨迹       
        center = (int((vBall[0] + vBall[2]) / 2), int((vBall[1] + vBall[3]) / 2))
        time_since_update[str(track_id)] = 0
        pts[track_id].append(center)

    for i in range(60):
        time_since_update[str(i)] += 1
        if(time_since_update[str(i)] >= 5):
            #pts[i].clear()
            if len(pts[i]) == 0:
                continue
            pts[i].popleft()
    
    for _id in range(60):
        colorT = COLORS[int(_id / 10) - 1]
        if len(pts[_id]) == 0:
            continue
        for j in range(len(pts[_id])):
            if pts[_id][j] is None:
                continue
            #thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            thickness = int(np.sqrt(32 + j * 6))
            #cv2.circle(frame, (pts[track_id][j]), 2, (color), thickness)
            cv2.circle(frame, (pts[_id][j]), thickness, (colorT),  -1)

    # 画人的ID()
    for kPerson, vPerson in personInfo.items():
        xx = int((vPerson[0] + vPerson[2]) / 2)
        yy = int((vPerson[1] + vPerson[3]) / 2)
        dx = (vPerson[2] - vPerson[0]) / 2
        dy = (vPerson[3] - vPerson[1]) / 2
        dist = np.sqrt(dx ** 2 +  dy ** 2)
        print(dist)
        
        # 透明操作
        

        colorG = COLORS[int(kPerson) - 1]
        cv2.circle(oriFrame, (xx, yy), int(dist), (colorG), -1)
        xTemp = (vPerson[0] * 4 + vPerson[2]) / 5
        yTemp = vPerson[1] + 5
#        if len(vPerson) > 4 and vPerson[4] != 0:
#            xTemp = vPerson[4]
#            yTemp = vPerson[5]
#        else:
#            llque = len(xpersonQue[int(kPerson)])
#            xTemp = xpersonQue[int(kPerson)][llque - 1]
#            yTemp = ypersonQue[int(kPerson)][llque - 1]
        xpersonQue[int(kPerson)].append(xTemp)
        ypersonQue[int(kPerson)].append(yTemp)
        xSum = 0
        ySum = 0
        # 平滑处理
        for _it in range(len(xpersonQue[int(kPerson)])):
            xSum = xSum + xpersonQue[int(kPerson)][_it]
            ySum = ySum + ypersonQue[int(kPerson)][_it]
        xText = xSum / len(xpersonQue[int(kPerson)])
        yText = ySum / len(xpersonQue[int(kPerson)])
        colorT = COLORS[int(kPerson) - 1]
        #cv2.putText(frame, kPerson, (int(xText), int(yText)), 0, 5e-3 * 250, (colorT), 3)
    
        
        # show pose point
#        if len(vPerson) > 4:
#            for _i in range(4, 54, 2):
#                cv2.circle(frame, (int(vPerson[_i]), int(vPerson[_i + 1])), 2, (colorT), 2)
                
#        # 画错误帧的pose点
#        thickness = 2
#        if frame_num == 346 and kPerson == '5' and len(vPerson) > 3:
#            frame_tmp = frame.copy()
#            cv2.line(frame, (int(vPerson[22]), int(vPerson[23])), (int(vPerson[24]), int(vPerson[25])), (color), thickness)
#            cv2.line(frame, (int(vPerson[24]), int(vPerson[25])), (int(vPerson[26]), int(vPerson[27])), (color), thickness)
#            cv2.line(frame, (int(vPerson[28]), int(vPerson[29])), (int(vPerson[30]), int(vPerson[31])), (color), thickness)
#            cv2.line(frame, (int(vPerson[30]), int(vPerson[31])), (int(vPerson[32]), int(vPerson[33])), (color), thickness)
#            #frame_num = frame_num - 1
#            for _ in range(20):
#                print(_)
#                out.write(frame) 
#                out.write(frame) 
#                out.write(frame) 
#                out.write(frame) 
#                out.write(frame) 
#                out.write(frame_tmp) 
#                out.write(frame_tmp) 
#                out.write(frame_tmp) 
#                out.write(frame_tmp) 
#                out.write(frame_tmp) 
#                
    img_mix = cv2.addWeighted(frame, 0.9, oriFrame, 0.1, 0)
    cv2.imshow("test", img_mix)
    out.write(frame) 
    frame_num = frame_num + 1
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
out.release()

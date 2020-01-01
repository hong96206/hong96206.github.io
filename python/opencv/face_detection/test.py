import numpy as np
import cv2
import sys
import os


# 人脸检测函数：
# useCamera参数：True表示使用摄像头，False表示读取当前目录下视频文件
def face_detection(useCamera=True):
    # namedWindow(winname[, flags]) -> None
    # .   @brief Creates a window.
    # 1.调用摄像头或者使用读取视频
    cv2.namedWindow("CaptureFace")
    if useCamera == True:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('ultraman.avi')

    # 2.人脸识别器分类器（GIT上开源的分类集）
    classfier = cv2.CascadeClassifier(r'./haarcascades' + os.sep + "haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)

    while cap.isOpened():
        flag, frame = cap.read()

        if not flag:
            break

        # 3.灰度转换
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4.人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 5.画图
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
        cv2.imshow("CaptureFace", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # face_detection()    # 使用摄像头
    face_detection(useCamera=False)  # 使用视频

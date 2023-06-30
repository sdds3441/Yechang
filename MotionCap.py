import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture('Video.mp4')

detector = PoseDetector()
posList = []
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img) #사람인식했는지 box 만들기

    if bboxInfo:
        lmString = '' # 포인트 저장
        for lm in lmList:
            lmString += f'{lm[1]}, {img.shape[0]-lm[2]}, {lm[3]}, ' #unity가 y축이 아래서부터 시작하기에 y축을 뒤집어줌

        posList.append(lmString)

    print(len(posList))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key ==ord('s'):
        with open('AnimationFile.txt', 'w') as f:
            f.writelines(["%s\n" % item for item in posList])

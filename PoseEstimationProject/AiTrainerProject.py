import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("AiTrainer/curls.mp4")

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

speed = 2
normal_fps = 30
delay = max(1, int(normal_fps / speed))

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1200, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Calculate angle and reps
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 100))
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 650),
                    cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 35)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

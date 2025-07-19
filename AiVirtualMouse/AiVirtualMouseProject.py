import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

frameR = 100 # frame reduction
wCam, hCam = 640, 480
pTime = 0
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wScr, hScr = autopy.screen.size()

detector = htm.handDetector(maxHands=1)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    # Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)
        # Only Index finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Both Index and Middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            # Find Distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # Click mouse if distance short
            if length < 30 :
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # FPS Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)












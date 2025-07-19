import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/1.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)

                print(id, x,y)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (20,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
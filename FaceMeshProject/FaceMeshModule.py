import cv2
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh


        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=True,  # better accuracy for lips & eyes
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )


        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,
                        self.drawSpec
                    )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (640, 360))
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS : {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

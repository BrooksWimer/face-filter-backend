import torch
import cv2
import numpy as np
from landmark_model import LandmarkCNN
from torchvision import transforms
from face import detect_faces_haar


class FaceLandmarkPredictor:
    def __init__(self, model_path, cascade_path=None, image_size=96):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LandmarkCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.image_size = image_size
        self.transform = transforms.ToTensor()
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, gray_image):
        return detect_faces_haar(gray_image, self.face_cascade)

    def select_face(self, faces, img_shape):
        center_x = img_shape[1] // 2
        center_y = img_shape[0] // 2

        def face_score(face):
            x, y, w, h = face
            cx, cy = x + w/2, y + h/2
            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            return distance - 0.01 * w * h

        return min(faces, key=face_score)

    def predict(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        if not faces:
            return None, None  # no face

        x, y, w, h = self.select_face(faces, gray.shape)
        face_crop = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_crop, (self.image_size, self.image_size))
        tensor = self.transform(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor).cpu().numpy().reshape(-1, 2)

        # Rescale to original image coordinates
        output *= [w, h]
        output += [x, y]
        return output, (x, y, w, h)

    def draw_landmarks(self, image, landmarks):
        for (px, py) in landmarks.astype(np.int32):
            cv2.circle(image, (px, py), 3, (0, 255, 0), -1)


if __name__ == "__main__":
    predictor = FaceLandmarkPredictor("landmark_model.pt")
    image = cv2.imread("test_photos/Ira.jpg")
    landmarks, bbox = predictor.predict(image)

    if landmarks is not None:
        predictor.draw_landmarks(image, landmarks)
        cv2.imshow("Prediction", image)
        cv2.waitKey(0)
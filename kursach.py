from fer import FER
import cv2
import matplotlib.pyplot as plt

test_img = cv2.imread('/Users/proxima/Downloads/test.jpg')
plt.imshow(test_img[:,:,::-1])

emo_detector = FER(mtcnn=True)
captured_emotions = emo_detector.detect_emotions(test_img)
print(captured_emotions)

dominant_emotion, emotion_score = emo_detector.top_emotion(test_img)
print(dominant_emotion, emotion_score)
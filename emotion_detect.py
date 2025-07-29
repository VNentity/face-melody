import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import time

model = load_model('/Users/vinodarava/Downloads/face melody/face_model.h5')
model.load_weights('/Users/vinodarava/Downloads/face melody/fece.weights.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

pygame.mixer.init()

music_files = {
    'angry': '/Users/vinodarava/Downloads/face melody/angry.mp3',
    'disgust': '/Users/vinodarava/Music/Music/Media.localized/Music/Devara Ayudha Pooja Bgm.mp3',
    'fear':'/Users/vinodarava/Downloads/face melody/fear.mp3',
    'happy': '/Users/vinodarava/Downloads/face melody/happy.mp3',
    'neutral': '/Users/vinodarava/Downloads/Fear.mp3',
    'sad': '/Users/vinodarava/Downloads/face melody/sad.mp3',
    'surprise': '/Users/vinodarava/Downloads/face melody/sup.mp3'
}

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
last_emotion = None
last_switch_time = time.time()
music_play_time = 10
last_music_time = 0

def play_music(emotion):
    global last_emotion, last_switch_time, last_music_time
    current_time = time.time()
    if (current_time - last_music_time) >= music_play_time:
        if emotion != last_emotion and (current_time - last_switch_time) >= 2:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(music_files[emotion])
            pygame.mixer.music.play(-1)
            last_emotion = emotion
            last_switch_time = current_time
            last_music_time = current_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))
        predictions = model.predict(face_resized)
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        play_music(emotion)

    cv2.imshow('Emotion Music Player', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()

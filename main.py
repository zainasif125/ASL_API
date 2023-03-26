import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
import numpy as np
import mediapipe
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense

model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(50, 258)),
    Dropout(0.2),
    LSTM(256, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(256, return_sequences=False, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights(r"E:\ASL\NewAction3.h5")


def prediction(video_content, lstm_model):
    actions = ['about', 'accident', 'africa', 'afternoon', 'again', 'all', 'always', 'animal', 'any', 'apple',
               'approve', 'argue', 'arrive', 'aunt', 'baby', 'back', 'bake', 'balance', 'bald', 'ball', 'banana', 'bar',
               'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom', 'before', 'better',
               'bicycle', 'bird', 'birthday', 'bitter', 'black', 'blue', 'book', 'both', 'bowl', 'bowling', 'box',
               'boy', 'boyfriend', 'bring', 'brother', 'brown', 'business', 'but', 'buy', 'can', 'candy', 'car',
               'cards', 'cat', 'catch', 'center', 'cereal', 'chair', 'change', 'check', 'cheese', 'chicken', 'children',
               'christmas', 'church', 'city', 'class', 'classroom', 'clock', 'clothes', 'coffee', 'cold', 'college',
               'color', 'computer', 'cook', 'cookie', 'cool', 'copy', 'corn', 'cough', 'country', 'cousin', 'cow',
               'crash', 'crazy', 'cute', 'dance', 'dark', 'day', 'deaf', 'decide', 'dentist', 'dictionary', 'different',
               'dirty', 'discuss', 'doctor', 'dog', 'doll', 'door', 'draw', 'drink', 'easy', 'eat', 'elevator', 'enjoy',
               'enter', 'environment', 'exercise', 'experience', 'face', 'family', 'far', 'fat', 'feel', 'fine',
               'finish', 'first', 'fish', 'fishing', 'food', 'football', 'forget', 'friend', 'friendly', 'from', 'full',
               'future', 'game', 'give', 'glasses', 'go', 'graduate', 'greece', 'green', 'hair', 'halloween', 'happy',
               'hat', 'have', 'headache', 'hearing', 'help', 'here', 'home', 'hospital', 'hot', 'house', 'how',
               'husband', 'interest', 'internet', 'investigate', 'jacket', 'jump', 'kiss', 'knife', 'know', 'language',
               'last', 'last year', 'later', 'law', 'learn', 'letter', 'library', 'like', 'list', 'lose', 'lunch',
               'magazine', 'man', 'many', 'match', 'mean', 'meat', 'medicine', 'meet', 'meeting', 'money', 'moon',
               'more', 'most', 'mother', 'movie', 'music', 'name', 'need', 'neighbor', 'nephew', 'never', 'nice',
               'niece', 'no', 'none', 'noon', 'north', 'not', 'now', 'nurse', 'off', 'ok', 'old', 'opinion', 'orange',
               'order', 'paint', 'paper', 'pencil', 'people', 'perspective', 'phone', 'pink', 'pizza', 'plan', 'play',
               'please', 'police', 'potato', 'practice', 'president', 'pull', 'purple', 'rabbit', 'rain', 'read', 'red',
               'remember', 'restaurant', 'ride', 'run', 'sad', 'school', 'science', 'secretary', 'sentence', 'share',
               'shirt', 'shoes', 'shop', 'sick', 'sign', 'since', 'sister', 'sleep', 'slow', 'small', 'smile', 'snow',
               'some', 'son', 'sorry', 'south', 'star', 'straight', 'strange', 'struggle', 'student', 'study', 'sunday',
               'suspect', 'sweetheart', 'table', 'tall', 'tea', 'teacher', 'tent', 'test', 'thank you', 'thanksgiving',
               'thin', 'think', 'thirsty', 'thursday', 'tiger', 'time', 'tired', 'toast', 'together', 'toilet',
               'tomorrow', 'traffic', 'travel', 'tree', 'truck', 'uncle', 'use', 'visit', 'wait', 'walk', 'want', 'war',
               'water', 'weak', 'wednesday', 'week', 'what', 'when', 'where', 'which', 'white', 'who', 'why', 'wife',
               'win', 'wind', 'window', 'with', 'woman', 'work', 'world', 'worry', 'write', 'wrong', 'year', 'yellow',
               'yes', 'yesterday', 'you']
    actions = actions[20:30]
    sequence = []
    cap = cv2.VideoCapture(video_content)

    mp_holistic = mediapipe.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        toskip = int(frame_count // 50)
        if toskip == 0:
            toskip = 1

        frame_num = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            frame_num = frame_num + toskip

            # rotate video right way up
            (h, w) = frame.shape[:2]
            rotpoint = (w // 2, h // 2)
            rotmat = cv2.getRotationMatrix2D(rotpoint, 180, 1.0)
            dim = (w, h)
            intermediateFrame = cv2.warpAffine(frame, rotmat, dim)

            # cropping
            size = intermediateFrame.shape
            finalFrame = intermediateFrame[80:(size[0] - 200), 30:(size[1] - 30)]

            # keypoint prediction
            image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False  # Image is no longer writeable
            results = holistic.process(image)  # Make prediction
            image.flags.writeable = True  # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

            # extract and append keypoints
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                21 * 3)
            keypoints = np.concatenate([pose, lh, rh])
            sequence.append(keypoints)

            if len(sequence) == 50:
                cap.release()
                break

    cap.release()
    cv2.destroyAllWindows()
    sequence = np.expand_dims(sequence, axis=0)[0]
    res = lstm_model.predict(np.expand_dims(sequence, axis=0))
    return actions[np.argmax(res)]


app = FastAPI()
origins = [
     "http://109.205.182.203:5420",
     ""
 ]

app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],)

@app.post("/")
async def read_root(file: UploadFile = File(...)):
    file_location = f"E:/ASL/videos/{file.filename}.mp4"
    with open(file_location, "wb") as video:
        video.write(await file.read())
    result=prediction(file_location,model)
    return {'Prediction':result}
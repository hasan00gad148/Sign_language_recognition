from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_WORD'] = 'static/files/word'
app.config['UPLOAD_SENTENCE'] = 'static/files/sentence'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit_word = SubmitField("Upload Word")
    submit_sentence = SubmitField("Upload Sentence")


words = ['Opaque', 'Red', 'Green', 'Yellow', 'Bright', 'Light-blue', 'Colors', 'Pink', 'Women', 'Enemy', 'Son', 'Man', 'Away', 'Drawer', 'Born', 'Learn', 'Call', 'Skimmer', 'Bitter', 'Sweet milk', 'Milk', 'Water', 'Food', 'Argentina', 'Uruguay', 'Country', 'Last name', 'Where', 'Mock', 'Birthday', 'Breakfast',
         'Photo', 'Hungry', 'Map', 'Coin', 'Music', 'Ship', 'None', 'Name', 'Patience', 'Perfume', 'Deaf', 'Trap', 'Rice', 'Barbecue', 'Candy', 'Chewing-gum', 'Spaghetti', 'Yogurt', 'Accept', 'Thanks', 'Shut down', 'Appear', 'To land', 'Catch', 'Help', 'Dance', 'Bathe', 'Buy', 'Copy', 'Run', 'Realize', 'Give', 'Find']
words = np.array(words)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1)


def get_model():
    global model
    # model = tf.keras.models.load_model(
    #     "/model/v8_64_5.5")
    reloaded_layer =tf.keras.layers.TFSMLayer(".\\API\\model\\v8_64_5.5", call_endpoint='serving_default')
    inputs = tf.keras.layers.Input(shape=(20, 170))  # Replace 'input_shape' with the actual shape
    # Call the TFSMLayer on the inputs
    outputs = reloaded_layer(inputs)

        # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    print(" Model Loaded DONE")


def select_words(result):
    max_length = 10
    max_count = 1
    sentence = []
    for i in range(len(result)):
        if i+1 >= len(result):
            if max_count >= max_length:
                if len(sentence) ==0:
                    sentence.append(result[i])
                elif sentence[-1]!= result[i]:
                    sentence.append(result[i])
            break

        if result[i] == result[i+1]:
            max_count += 1
        else:
            if max_count >= max_length:
                if len(sentence) ==0:
                    sentence.append(result[i])
                elif sentence[-1]!= result[i]:
                    sentence.append(result[i])
            max_count = 1
    return sentence


def predictUsingWord(v):
    p = model.predict(np.reshape(v, (1, 20, 170)))
    print(np.argmax(p['dense']),words[np.argmax(p['dense'])])
    return words[np.argmax(p['dense'])]


def predictUsingSentence(v):
    result = []
    v = np.reshape(v, (1, v.shape[0], 170))
    for i in range(0, v.shape[1]-20):
        p = model.predict(v[:, i:i+20, :])
        result.append(words[np.argmax(p)])
    
    sentence = select_words(result)
    FinalResult = ' '.join(sentence)
    return FinalResult


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    return results


def normalize_zscore(pose, face, lh, rh):
    pose = (pose - pose.mean(axis=0))/(pose.std(axis=0) + 1e-7)
    face = (face - face.mean(axis=0))/(face.std(axis=0) + 1e-7)
    lh = (lh - lh.mean(axis=0))/(lh.std(axis=0) + 1e-7)
    rh = (rh - rh.mean(axis=0))/(rh.std(axis=0) + 1e-7)

    return pose, face, lh, rh


def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]
                    ) if results.pose_landmarks else np.zeros((33, 2))
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]
                    ) if results.face_landmarks else np.zeros((468, 2))
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]
                  ) if results.left_hand_landmarks else np.zeros((21, 2))
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]
                  ) if results.right_hand_landmarks else np.zeros((21, 2))

    pose, face, lh, rh = normalize_zscore(pose, face, lh, rh)
    face = face[:10, :]

    return np.concatenate([face.flatten(), pose.flatten(), lh.flatten(), rh.flatten()])


def videoProc2(path, c=1, sentence=False):
    print(path)
    cap = cv2.VideoCapture(path)
    success = True
    framecount = 20

    fnum = 0
    v = []
    fpsCounter = 0

    while success:
        success, frame = cap.read()

        if fpsCounter <= 0:
            fpsCounter = c
        else:
            fpsCounter -= 1
            continue
        if fnum == framecount and sentence == False:
            break

        if success:
            # frame = cv2.resize(frame, (720,480), interpolation = cv2.INTER_AREA)
            results = mediapipe_detection(frame, holistic)

            eres = extract_keypoints(results)
            tmp = np.reshape(eres, (1, -1))
            v.append(tmp)

            fnum += 1

    v = np.reshape(v, (fnum, -1)) if fnum > 0 else []
    if fnum < framecount and fnum > 0:
        tmp = np.zeros((framecount-fnum, 170))
        v = np.concatenate((v, tmp), axis=0)

    return v


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        configKey = "UPLOAD_WORD"
        typeOfVideo = "word"
        file = form.file.data  # First grab the file
        if form.submit_sentence.data:
            configKey = "UPLOAD_SENTENCE"
            typeOfVideo = "sentence"

        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                  app.config[configKey], secure_filename(file.filename)))  # Then save the file
        videoPath = os.path.abspath(os.path.dirname(
            __file__))+"/"+app.config[configKey]+"/"+file.filename.replace(" ","_")

        get_model()
        if typeOfVideo == "word":
            v = videoProc2(videoPath, c=2)
            return render_template('index.html', form=form, text=predictUsingWord(v))
        else:
            v = videoProc2(videoPath, c=1, sentence=True)
            return render_template('index.html', form=form, text=predictUsingSentence(v))

    return render_template("index.html",form=form)


if __name__ == '__main__':
    app.run(debug=True,port=3300)

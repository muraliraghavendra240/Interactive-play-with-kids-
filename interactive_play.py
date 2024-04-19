from gtts import gTTS
import random
import pygame
import time

def speech_to_text():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
    # print("Please speak something...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        start()
    except sr.RequestError:
        start()

def poems():
    rhymes = ['Baa baa black sheep.mp3','Ding Dong Bell.mp3','Hickory Dickory Dock.mp3','Humpty Dumpty.mp3',"I_m A Little Teapot.mp3",'Itsy bitsy spider.mp3','Jack And Jill.mp3','Johny Johny Yes Papa.mp3','London Bridge Is Falling Down.mp3','Old MacDonald.mp3','PussyCat, PussyCat.mp3','Rain Rain Go away.mp3','Star Light Star Bright.mp3','The Alphabet Song.mp3','Three Blind Mice.mp3','Twinkle Twinkle Little Star.mp3','Wheels on the Bus.mp3']
    rhyme = random.choice(rhymes)
    content='/home/raspberrypi/Desktop/rhymes/'
    path = content + rhyme
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.quit()

def stories():
    stories=['Raman The detective.mp3','The bear and the two friends.mp3','The boy who cried wolf.mp3','The dog at the well.mp3','The fox and the grapes.mp3','The golden egg.mp3','The golden touch.mp3','The miser and his gold.mp3','The proud rose.mp3','The wise old owl.mp3']
    content='/home/raspberrypi/Desktop/stories_audio/'
    story = random.choice(stories)
    path = content + story
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.quit()

def conversation(query):
    import pandas as pd
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    df = pd.read_csv('/home/raspberrypi/Desktop/answer.csv',encoding="ISO-8859-1")
    X = df['query']
    y = df['text']
    label_to_id = {text: i for i, text in enumerate(y.unique())}
    y = y.map(label_to_id)
    max_sequence_length = 50
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    loaded_model = tf.keras.models.load_model('/home/raspberrypi/Desktop/LSTM_94_97.h5')
    def classify_conversation(text):
        text_sequence = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_sequence, maxlen=max_sequence_length, padding='post', truncating='post')
        prediction = loaded_model.predict(text_padded)
        predicted_label = np.argmax(prediction)
        return predicted_label

    predicted_label = classify_conversation(query)
    print(f"Predicted Label: {predicted_label}")
    id_to_label = {i: label for label, i in label_to_id.items()}
    predicted_label_id = classify_conversation(query)
    predicted_label_text = id_to_label[predicted_label_id]
    tts = gTTS(predicted_label_text)
    tts.save('output.mp3')
    pygame.init()
    pygame.mixer.music.load('output.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.quit()

def start():
    wake_word= speech_to_text()
    if "hello" in wake_word:
        pygame.init()
        pygame.mixer.music.load("/home/raspberrypi/Desktop/wake.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        pygame.quit()
        text = speech_to_text()
        if "play rhymes" in text:
            poems()
            start()
        elif "tell stories" in text:
            stories()
            start()
        else:
            conversation(text)
            start()
    else:
        start()


 
pygame.init()
pygame.mixer.music.load("/home/raspberrypi/Desktop/intro.mp3")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    time.sleep(1)
pygame.quit()

start()

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


class DataPreprocessor:
    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        pixels = df['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        emotions = pd.get_dummies(df['emotion']).values
        return faces, emotions
class EmotionDetector:
    def __init__(self, data_path, image_size, num_classes, batch_size, epochs):
        self.data_path = data_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_preprocessor = DataPreprocessor(data_path, image_size)
        self.model = self.build_model()
        
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu', input_shape=(48,48,1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model
        
    def train(self):
        faces, emotions = self.data_preprocessor.load_data()
        faces = np.expand_dims(faces, -1)
        x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
        datagen = ImageDataGenerator(horizontal_flip=True)
        datagen.fit(x_train)
        history = self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                              epochs=self.epochs,
                              verbose=1,
                              validation_data=(x_test, y_test),
                              shuffle=True)
        self.model.save('emotion_model.h5')
        return history
detector = EmotionDetector(data_path='fer2013.csv', image_size=(48, 48), num_classes=7, batch_size=64, epochs=30)
history = detector.train()

test_faces, test_emotions = detector.data_preprocessor.load_data()
test_faces = np.expand_dims(test_faces, -1)
predictions = detector.model.predict(test_faces)
predicted_emotions = np.argmax(predictions, axis=1)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(test_emotions, axis=1), predicted_emotions)
print(cm)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(np.argmax(test_emotions, axis=1), predicted_emotions))


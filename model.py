import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

train_folder = "/Users/proxima/Documents/archive/train"
test_folder = "/Users/proxima/Documents/archive/test"

def load_data(folder):
    X = []
    y = []
    emotions = []
    label = 0
    for dir in sorted(os.listdir(folder)):
        path = os.path.join(folder, dir)
        if os.path.isdir(path):
            emotions.append(dir)
            for filename in os.listdir(path):
                if filename.lower().endswith('.jpg'):
                    img = cv2.imread(os.path.join(path, filename))
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(gray, (48, 48))
                        img = np.array(img) / 255.0
                        X.append(img)
                        y.append(label)
            label += 1
    return np.array(X).reshape(-1, 48, 48, 1), np.array(y), emotions

X_train, y_train, train_emotions = load_data(train_folder)
X_test, y_test, test_emotions = load_data(test_folder)

model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, (3, 3), padding='same', activation='elu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='elu'),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),
    Conv2D(64, (3, 3), padding='same', activation='elu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='elu'),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='elu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='elu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='elu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Увеличиваем количество выходов до 7 для соответствия классам
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/Users/proxima/Documents/model/Emotion_7.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)
callbacks = [checkpoint, reduce_lr]

history = model.fit(X_train, y_train, batch_size=32, epochs=23, validation_data=(X_test, y_test), callbacks=callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy during training and validation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери на этапах проверки и обучения')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Потери на этапе обучения', 'Потери на этапе проверки'], loc='upper left')
plt.show()

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
# читаем изображения из папки
images = [cv2.imread(file) for file in glob.glob(r"/Users/proxima/Documents/Kursach/faces/*")]
trained_face_data = cv2.CascadeClassifier(r'/Users/proxima/Documents/Kursach/haarcascade_frontalface_default.xml')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))
font = cv2.FONT_HERSHEY_TRIPLEX

# в цикле просматриваем все изображения
for i in range(len(images)):
    # переводим изображение в градации серого
    grayscaled_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    # определяем координаты лица
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        # рисуем рамку вокруг лица
        cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 255, 0), 15)

        # изменяем размер изображения и нормируем значения
        roi_gray = grayscaled_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # делаем прогноз и определяем класс
            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            # над рамкой пишем эмоцию, предсказанную классификатором
            cv2.putText(images[i], label, label_position, font, 3, (0, 0, 255), 7)
        else:
            cv2.putText(images[i], 'No Face Found', (30, 60), font, 2, (0, 0, 255), 3)

    # отображаем изображения
    ax[i].axis("off")
    ax[i].imshow(images[i][:, :, ::-1])

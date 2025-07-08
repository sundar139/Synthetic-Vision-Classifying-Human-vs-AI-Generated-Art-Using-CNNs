import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import time

top_dir = 'Dataset'
train_dir = os.path.join(top_dir, 'train')
test_dir = os.path.join(top_dir, 'test')

train_human = [os.path.join(train_dir, directory) for directory in os.listdir(train_dir) if not directory.startswith('AI_')]
train_ai = [os.path.join(train_dir, directory) for directory in os.listdir(train_dir) if directory.startswith('AI_')]
test_human = [os.path.join(test_dir, directory) for directory in os.listdir(test_dir) if not directory.startswith('AI_')]
test_ai = [os.path.join(test_dir, directory) for directory in os.listdir(test_dir) if directory.startswith('AI_')]

train_data = pd.DataFrame(columns=['filepath', 'label'])
test_data = pd.DataFrame(columns=['filepath', 'label'])

for directory in train_human + train_ai:
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        label = "human" if directory in train_human else "AI"
        train_data = train_data._append({'filepath': filepath, 'label': label}, ignore_index=True)

for directory in test_human + test_ai:
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        label = "human" if directory in test_human else "AI"
        test_data = test_data._append({'filepath': filepath, 'label': label}, ignore_index=True)

print(train_data.head())

random_seed = 24
np.random.seed(random_seed)
num_to_drop = len(train_data[train_data['label'] == 'AI']) - 55015
ai_indices = train_data[train_data['label'] == 'AI'].index
indices_to_drop = np.random.choice(ai_indices, num_to_drop, replace=False)
train_data = train_data.drop(indices_to_drop).reset_index(drop=True)

print(train_data['label'].value_counts())

print(test_data.head())

training_generator = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)
test_generator = ImageDataGenerator(rescale=1./255)

train_dataset = training_generator.flow_from_dataframe(
    dataframe=train_data,
    x_col='filepath',
    y_col='label',
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical',  
    shuffle=True
)

test_dataset = test_generator.flow_from_dataframe(
    dataframe=test_data,
    x_col='filepath',
    y_col='label',
    target_size=(32, 32),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

network = Sequential()
network.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu'))
network.add(MaxPooling2D())
network.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
network.add(MaxPooling2D())
network.add(Flatten())
network.add(Dense(units=64, activation='relu'))
network.add(Dense(units=2, activation='softmax'))


network.summary()
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
total_training_time = 0

for epoch in range(epochs):
    start_time = time.time()
    history = network.fit_generator(train_dataset, epochs=1, validation_data=test_dataset, callbacks=[model_checkpoint])
    training_time = time.time() - start_time
    total_training_time += training_time
    print("Epoch {} - Total training time so far: {:.2f} seconds".format(epoch + 1, total_training_time))

    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.show()

    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.show()

    predictions = network.predict(test_dataset)
    predictions = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_dataset.classes, predictions)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(test_dataset.classes, predictions))

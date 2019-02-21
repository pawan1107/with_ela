import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import img_to_array
import cv2
import sys
import os

def main():
    dataset = pd.read_csv('dataset.csv')
    X = []
    Y = []
    for ind, row in dataset.iterrows():
        img = cv2.imread(row[1])
        img = cv2.resize(img, (256, 256))
        img = img_to_array(img)
        X.append(img)
        Y.append(row[2])

    X = np.array(X, dtype="float") / 255.0
    Y = to_categorical(Y, 2)
    Y = np.array(Y)  

    X = X.reshape(-1, 256, 256, 3)

    # Split Train test
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)


    cnn_model=Sequential()

    cnn_model.add(
        Conv2D(
            input_shape=(256, 256, 3), 
            filters=64, 
            kernel_size=(3, 3), 
            padding='valid',
            activation='relu',
            ))

    cnn_model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3), 
            padding='valid', 
            activation='relu', 
            ))
            
    cnn_model.add(
        MaxPool2D(
            pool_size=(2, 2), 
            ))


    cnn_model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3), 
            padding='valid', 
            activation='relu', 
            ))
            
    cnn_model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3), 
            padding='valid', 
            activation='relu', 
            ))

    cnn_model.add(
        MaxPool2D(
            pool_size=2, 
            ))


    cnn_model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3), 
            padding='valid', 
            activation='relu', 
            ))

    cnn_model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3), 
            padding='valid', 
            activation='relu', 
            ))

    cnn_model.add(
        MaxPool2D(
            pool_size=2, 
            ))

    # cnn_model.add(
    #     Conv2D(
    #         filters=512,
    #         kernel_size=(3, 3), 
    #         padding='valid', 
    #         activation='relu', 
    #         ))

    # cnn_model.add(
    #     Conv2D(
    #         filters=512,
    #         kernel_size=(3, 3), 
    #         padding='valid', 
    #         activation='relu', 
    #         ))

    # cnn_model.add(
    #     MaxPool2D(
    #         pool_size=2,
    #         ))

    cnn_model.add(
        Conv2D(
            filters=20, 
            kernel_size=(4, 4), 
            padding='valid', 
            activation='relu',
            ))

    cnn_model.add(
        MaxPool2D(
            pool_size=2, 

            ))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(256, activation = "relu"))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(2, activation = "softmax"))

    print(cnn_model.summary())

    cnn_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
        )
    

    early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=0, mode='auto')

    saving_weight = ModelCheckpoint(
    	'weights{epoch:08d}.h5',
    	save_weights_only=True,
    	period=5)


    epochs = 100
    batch_size = 32
    
    H = cnn_model.fit(
        X_train, 
        Y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping, saving_weight])

    cnn_model.save('keras_cnn_model_redone.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('model.png')


if __name__=='__main__':
    main()

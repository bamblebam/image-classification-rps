# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# %%
train_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
gen_train = train_gen.flow_from_directory(
    "datasets/rps", class_mode='sparse', batch_size=64, color_mode='grayscale', target_size=(64, 64))
# %%
test_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
gen_test = test_gen.flow_from_directory(
    "datasets/rps-test-set", class_mode='sparse', batch_size=64, color_mode='grayscale', target_size=(64, 64))

# %%
model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 1), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
model.summary()
# %%
reduceLR = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', min_delta=0.001, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'models/model7.h5', save_best_only=True, monitor='val_loss', mode='min')
callbacks = [reduceLR, earlyStop, checkpoint]
# %%
model.fit(x=gen_train, epochs=100,
          validation_data=gen_test, callbacks=callbacks)
# %%
score = model.evaluate(gen_test)
print(score[1])
# %%
model.save('models/final_model.h5')
# %%

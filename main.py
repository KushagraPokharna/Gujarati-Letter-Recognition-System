import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

# Define CNN model (Convolutional Neural Network)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=385, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # used optimizer to optimize

# Display the model summary
model.summary()


# Define data generators for training and testing
# train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1.0/255,
                                  zoom_range=0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(r'E:\ML PROJECT\Train',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=True)

test_datagen = ImageDataGenerator(rescale=1.0/255,
                                  zoom_range=0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  fill_mode='nearest')
test_generator = test_datagen.flow_from_directory(r'E:\ML PROJECT\Test',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=True)

# Specify the path to your dataset
# train_generator = train_datagen.flow_from_directory(r'F:\MIT-WPU\New folder\Gujarati OCR\Gujarati\Train', target_size=(64, 64), batch_size=32, class_mode='categorical')
# test_generator = test_datagen.flow_from_directory(r'F:\MIT-WPU\New folder\Gujarati OCR\Gujarati\Test', target_size=(64, 64), batch_size=32, class_mode='categorical')


# Train the model
model.fit(train_generator,steps_per_epoch=train_generator.samples // 32, epochs=50, validation_data=test_generator,validation_steps=test_generator.samples // 32) #creating batches by dividing total data/32


'''
import time
# !pip install -q pyyaml h5py
# Required to save models in HDF5
format
now = time.strftime("%Y%m%d%H%M%S", time.localtime())
filepath = r'E:/ML PROJECT'
callback_logger = CSVLogger( filepath + '/' +
"log_training_m4{}.csv".format(now)
 , separator=','
 , append=False
 )
best_model_file = "m4({val_accuracy}).h5"
callack_saver = tf.keras.callbacks.ModelCheckpoint(
 filepath + '/' + best_model_file
 , monitor='val_accuracy'
 , verbose=1
 , mode='max'
 , period=1
 ,save_best_only = True
 )


list_callback = [
 callback_logger
 ,callack_saver]
'''
model.save(r"E:/ML PROJECT/tf_custom_test.h5")

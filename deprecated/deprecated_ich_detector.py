#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:26:31 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
from keras import preprocessing, models, layers

from matplotlib import pyplot
import numpy

############################## Custom Modules ##############################
from preprocessing import process_dataset

############################## Config ##############################
import config

############################## Constants ##############################


############################## Debug Constants ##############################


############################## Execute Preprocessing Pipeline ##############################
def main():
    """
    I.C.H. detector entry point.
    """
    processed_data = process_dataset()
    (train_images, train_labels) = processed_data[0]
    (test_images, test_labels) = processed_data[1]
    (val_images, val_labels) = processed_data[2]
    (images, d_raw_images, d_brain_images, d_gray_images) = processed_data[3]
    dataset_len = len(images)

    #print(*train_images)
    #print(*train_labels)
    #print(*test_images)
    #print(*test_labels)
    #print(*val_images)
    #print(*val_labels)
    #print(len(images))

    #Create model.
    siim_model = models.Sequential()

    #Base shape
    siim_model.add(layers.Conv2D(config.CONVOLUTION_LAYERS_FILTERS, (3, 3), input_shape=config.INPUT_SHAPE))
    siim_model.add(layers.Activation(config.ACTIVATION_LAYER))

    #Reduce param count.
    siim_model.add(layers.MaxPool2D(pool_size=config.CONVOLUTION_LAYERS_POOL_SIZE))

    #Finer processing.
    for _ in range(config.CONVOLUTION_LAYERS_COUNT - 1):
        siim_model.add(layers.Conv2D(config.CONVOLUTION_LAYERS_FILTERS, (3, 3)))
        siim_model.add(layers.Activation(config.ACTIVATION_LAYER))
        siim_model.add(layers.MaxPool2D(pool_size=config.CONVOLUTION_LAYERS_POOL_SIZE))

    #Dense layers.
    siim_model.add(layers.Flatten())
    siim_model.add(layers.Dense(config.FULL_LAYERS_DENSITY))
    siim_model.add(layers.Activation(config.ACTIVATION_LAYER))

    #Reduce overfitting.
    siim_model.add(layers.Dropout(config.DROPOUT_COEFFICIENT))

    #Prepare and grab output.
    siim_model.add(layers.Dense(config.OUTPUT_DENSITY))
    siim_model.add(layers.Activation(config.OUTPUT_ACTIVATION_LAYER))

    #Compile model.
    siim_model.compile(loss=config.LOSS,
            optimizer=config.OPTIMIZER,
            metrics=["accuracy"],
            run_eagerly=config.D_RUN_EAGERLY)

    #Prepare training data.
    train_data_generator = preprocessing.image.ImageDataGenerator(rescale=config.RESCALE,
            shear_range=config.SHEAR,
            zoom_range=config.ZOOM,
            rotation_range=config.ROTATION,
            width_shift_range=config.SHIFT_RANGE,
            height_shift_range=config.SHIFT_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP)

    train_generator = train_data_generator.flow(train_images[..., numpy.newaxis],
            train_labels,
            batch_size=config.BATCH_SIZE)

    #Prepare validation data.
    val_data_generator = preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    val_generator = val_data_generator.flow(val_images[..., numpy.newaxis],
            val_labels,
            batch_size=config.BATCH_SIZE)

    #Begin training!
    history = siim_model.fit_generator(train_generator,
            steps_per_epoch=len(train_images) // config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=val_generator,
            validation_steps=len(val_images // config.BATCH_SIZE))


    # Plot training & validation accuracy values
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('Model accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.show()

    # Plot training & validation loss values
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.show()

if __name__ == "__main__":
    main()
    
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
# Image_Classifier Function
# TODO Update classifier using ResNet50 and Inception
# TODO Calculate metrics: accuracy, sensitiviy, recall, performance


def image_classifier():
    num_classes = 2

    my_new_model = Sequential()
    my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
    my_new_model.add(Dense(num_classes, activation='softmax'))

    # Indicate whether the first layer should be trained/changed or not.
    my_new_model.layers[0].trainable = False

    print("Specify Model Complete")

    my_new_model.compile(optimizer='sgd',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    print("Compile Model Complete")

    image_size = 224
    data_generator = ImageDataGenerator(preprocess_input, validation_split=0.2)

    path = Path("images")

    train_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(image_size, image_size),
        batch_size=10,
        class_mode='categorical',
        subset="training")

    validation_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(image_size, image_size),
        class_mode='categorical',
        subset="validation")

    # fit_stats below saves some statistics describing how model fitting went
    # the key role of the following line is how it changes my_new_model by fitting to data
    fit_stats = my_new_model.fit_generator(train_generator,
                                           steps_per_epoch=1,
                                           epochs=10,
                                           validation_data=validation_generator,
                                           validation_steps=1)
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S

    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    print("date and time =", dt_string)
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(fit_stats.history['acc'])
    axs[0].plot(fit_stats.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Test'], loc='upper left')
    fig.suptitle('Model Run ' + dt_string, fontsize=16)

    axs[1].plot(fit_stats.history['loss'])
    axs[1].plot(fit_stats.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Train', 'Test'], loc='upper left')

    plt.show()

    print("Fit Model Complete")

# U-Net Semantic Segmentation Function
# TODO Create U-Net Lesion Semantic Segmentator
# def u_net_lesion():

image_classifier()
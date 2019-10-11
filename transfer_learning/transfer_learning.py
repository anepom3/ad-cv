import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
# Image_Classifier Function
# TODO Update classifier using ResNet50 and Inception
# TODO Calculate metrics: accuracy, sensitiviy, recall, performance
def image_classifier():
    num_classes = 2
    # resnet_weights_path = '../Models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

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
    data_generator = ImageDataGenerator(preprocess_input)

    train_path = Path(
        "C:\\Users\\antho\\Google Drive\\anepom3\\UIUC\\Extracurriculars\\BMES\\Design Team\\Health Mirror\\Software\\Transfer Learning\\AD\\images\\train")
    validation_path = Path(
        "C:\\Users\\antho\\Google Drive\\anepom3\\UIUC\\Extracurriculars\\BMES\\Design Team\\Health Mirror\\Software\\Transfer Learning\\AD\\images\\val")

    train_generator = data_generator.flow_from_directory(
        directory=train_path,
        target_size=(image_size, image_size),
        batch_size=10,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        directory=validation_path,
        target_size=(image_size, image_size),
        class_mode='categorical')

    # fit_stats below saves some statistics describing how model fitting went
    # the key role of the following line is how it changes my_new_model by fitting to data
    for i in range(10):
        fit_stats = my_new_model.fit_generator(train_generator,
                                               steps_per_epoch=22,
                                               validation_data=validation_generator,
                                               validation_steps=1)
        print(fit_stats.history.keys())
        print(fit_stats.history['val_acc'][0])

    print("Fit Model Complete")

# U-Net Semantic Segmentation Function
# TODO Create U-Net Lesion Semantic Segmentator
# def u_net_lesion():

image_classifier()
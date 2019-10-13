import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# Image_Classifier Function
# TODO Update classifier using ResNet50 and Inception
# TODO Calculate metrics: accuracy, sensitiviy, recall, performance


def image_classifier():
    num_classes = 2
    np.random.seed(seed=7)
    for test_epochs in range(10, 11, 10):
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

        # checkpoint
        filepath = "weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        image_size = 299
        seed = 7
        data_generator = ImageDataGenerator(preprocess_input, validation_split=0.2)
        aug_data_generator = ImageDataGenerator(preprocess_input,
                                                validation_split=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True)
        path = Path("images")

        train_generator = aug_data_generator.flow_from_directory(
            directory=path,
            target_size=(image_size, image_size),
            batch_size=10,
            class_mode='categorical',
            subset="training",
            seed=seed)

        validation_generator = data_generator.flow_from_directory(
            directory=path,
            target_size=(image_size, image_size),
            batch_size=10,
            class_mode='categorical',
            subset="validation",
            seed=seed)

        # fit_stats below saves some statistics describing how model fitting went
        # the key role of the following line is how it changes my_new_model by fitting to data
        fit_stats = my_new_model.fit_generator(train_generator,
                                               steps_per_epoch=17,
                                               epochs=test_epochs,
                                               validation_data=validation_generator,
                                               validation_steps=5,
                                               callbacks=callbacks_list,
                                               verbose=2)
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S

        dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
        save_date = now.strftime("%m_%d_%y_%H_%M_%S")
        save_fig_path = Path('results/Graphs') / (save_date + '_AD.png')
        save_result_path = Path('results/Tables/result_log.csv')
        print(save_fig_path)
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
        plt.savefig(save_fig_path)
        # plt.show()

        import csv
        val_acc = fit_stats.history['val_acc']
        acc = fit_stats.history['acc']
        val_loss = fit_stats.history['val_loss']
        loss = fit_stats.history['loss']
        with open(save_result_path, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([dt_string]+val_acc+acc+val_loss+loss)
        csvFile.close()

        print("Fit Model Complete")

# U-Net Semantic Segmentation Function
# TODO Create U-Net Lesion Semantic Segmentator
# def u_net_lesion():


image_classifier()

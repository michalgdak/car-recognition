'''
Created on 22 sty 2018

@author: mgdak
'''

import os
import argparse
import sys
from pprint import pprint
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
import shutil as sh
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras import regularizers

train_data_dir = 'train'
val_data_dir = 'val'
nb_train_samples = 8144
nb_val_samples = 8041

#Creates an array with following values: picture name, picture category ID, train/validation label       
def readData(matFile):   
    content = sio.loadmat(matFile)
    data = [(_[0][0][:],_[5][0][0],_[6][0][0]) for _ in content['annotations'][0]]
    return data

#Creates an array of all classes
def readClasses(matFile):   
    content = sio.loadmat(matFile)
    classes = [(_[0]) for _ in content['class_names'][0]]
    return classes

#Movces raw data (pictures) into respective category subfolders with train/validation division 
def dataPreprocessing(dataDir, labelsFile):
    data = readData(labelsFile)
    classes = readClasses(labelsFile)
    print("---------------")
    for recData in data:
        if recData[2] == 1:
            #validation set
            os.makedirs(dataDir + "/" + val_data_dir + "/" + classes[recData[1] - 1] + "/", exist_ok=True)
            sh.move(dataDir + "/" + recData[0][8:], dataDir + "/" + val_data_dir + "/" + classes[recData[1] - 1] + "/" + recData[0][8:])
        else:
            os.makedirs(dataDir + "/" + train_data_dir + "/" + classes[recData[1] - 1] + "/", exist_ok=True)
            sh.move(dataDir + "/" + recData[0][8:], dataDir + "/" + train_data_dir + "/" + classes[recData[1] - 1] + "/" + recData[0][8:]) #train set

#serializes the trained model and its weights
def serializeModel(model, fileName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")

def model(learningRate, optimazerLastLayer, noOfEpochs, batchSize, savedModelName, srcImagesDir, labelsFile):
    
    classes = readClasses(labelsFile)
    
    # this is the augmentation configuration used for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True)
    
    # this is the augmentation configuration used for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    # this is a generator that will read pictures found in
    # subfolers of 'car_ims_dir/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        srcImagesDir + "/" + train_data_dir + "/", # this is the target directory
        target_size=(299, 299), # all images will be resized to 299x299
        batch_size=batchSize, 
        class_mode='categorical') # since we use categorical_crossentropy loss, we need categorical labels
   
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        srcImagesDir + "/" + val_data_dir + "/", 
        target_size=(299, 299), 
        batch_size=batchSize, 
        class_mode='categorical')

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01))(x)
    # add Dropout regularizer
    x.add(Dropout(0.5))
    # and a logistic layer with all car classes
    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01))(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimazerLastLayer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batchSize,
        epochs=noOfEpochs,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batchSize)
    
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    
    plt.savefig(savedModelName + '_initialModel_plot.png')
    
    serializeModel(model, savedModelName + "_initialModel")
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=learningRate, momentum=0.9), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batchSize,
        epochs=noOfEpochs,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batchSize)
    
    plt.clf()
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.savefig(savedModelName + '_finalModel_plot.png')
    
    serializeModel(model, savedModelName + "_finalModel")

def main(args):
    pprint(args)
    if args.process_data:
        dataPreprocessing(args.car_ims_dir, args.car_ims_labels)
    
    model(args.learning_rate, 
          args.optimizer_last_layer, 
          args.no_of_epochs, 
          args.batch_size, 
          args.saved_model_name,
          args.car_ims_dir, 
          args.car_ims_labels)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--process_data', 
        help='Divides the whole data set from Stanford (http://ai.stanford.edu/~jkrause/cars/car_dataset.html) into test and validation sets',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=True)
        
    parser.add_argument('--car_ims_dir', type=str, 
        help='Directory where all pictures are located or where subfolder train/val are located', default='~/car_ims')

    parser.add_argument('--car_ims_labels', type=str, 
        help='Points to the file with all labels', default='~/cars_annos.mat')    

    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0001)
    
    parser.add_argument('--optimizer_last_layer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='RMSPROP')

    parser.add_argument('--no_of_epochs', type=int,
        help='Number of epochs to run.', default=500)

    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=128)
    
    parser.add_argument('--saved_model_name', type=str,
        help='Number of images to process in a batch.', default='carRecognition')
    
    return parser.parse_args(argv)
    
if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
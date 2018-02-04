'''
Created on 22 sty 2018


3D Object Representations for Fine-Grained Categorization
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

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
from keras.applications.densenet import DenseNet121
from keras import callbacks
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras import regularizers
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K

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


def prepareDataGenerators(batchSize, srcImagesDir, labelsFile):
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
        target_size=(224, 224), # all images will be resized to 299x299
        batch_size=batchSize, 
        class_mode='categorical') # since we use categorical_crossentropy loss, we need categorical labels
# this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        srcImagesDir + "/" + val_data_dir + "/", 
        target_size=(224, 224), 
        batch_size=batchSize, 
        class_mode='categorical')
    return classes, train_generator, validation_generator


def getVGG16Architecture(classes, dropoutRate):
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #for layer in enumerate(base_model.layers):
    #    layer[1].trainable = False
    
    #flatten the results from conv block
    x = Flatten()(base_model.output)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    #add logistic layer with all car classes
    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def getVGG19Architecture(classes, dropoutRate):
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in enumerate(base_model.layers):
        layer[1].trainable = False
    
    #flatten the results from conv block
    x = Flatten()(base_model.output)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)

    #add logistic layer with all car classes
    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def getInceptionV3Architecture(classes, dropoutRate):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # InceptionV3
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    x = Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01))(x)
    
    # add Dropout regularizer
    x = Dropout(dropoutRate)(x)
    
    # and a logistic layer with all car classes
    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in enumerate(base_model.layers):
        layer[1].trainable = False
    
    return model


def setLayersToRetrain(model, modelArchitecture):
    
    if modelArchitecture == 'InceptionV3':
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:249]:
            layer.trainable = False
        
        for layer in model.layers[249:]:
            layer.trainable = True
    elif modelArchitecture == 'VGG16':
        #train the last conv block
        #for layer in model.layers[:15]:
         #   layer.trainable = False
        
        for layer in model.layers[15:]:
            layer.trainable = True
    elif modelArchitecture == 'VGG19':
        #train the last conv block
        for layer in model.layers[:17]:
            layer.trainable = False
        
        for layer in model.layers[17:]:
            layer.trainable = True
        


def initialTraining(optimazerLastLayer, noOfEpochs, batchSize, savedModelName, train_generator, validation_generator, model):
    # compile the model and train the top layer only
    model.compile(optimizer=optimazerLastLayer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=nb_train_samples // batchSize, 
        epochs=noOfEpochs, 
        validation_data=validation_generator, 
        validation_steps=nb_val_samples // batchSize,
        callbacks=[earlystop])
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.savefig(savedModelName + '_initialModel_plot.png')
    serializeModel(model, savedModelName + "_initialModel")


def finetuningTraining(learningRate, noOfEpochs, batchSize, savedModelName, train_generator, validation_generator, model):
    # we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=learningRate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=nb_train_samples // batchSize, 
        epochs=noOfEpochs, 
        validation_data=validation_generator, 
        validation_steps=nb_val_samples // batchSize,
        callbacks=[earlystop])
    plt.clf()
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.savefig(savedModelName + '_finalModel_plot.png')
    serializeModel(model, savedModelName + "_finalModel")


def model(learningRate, optimazerLastLayer, noOfEpochs, batchSize, savedModelName, srcImagesDir, labelsFile, modelArchitecture, dropoutRate):
    
    classes, train_generator, validation_generator = prepareDataGenerators(batchSize, srcImagesDir, labelsFile)

    if modelArchitecture == 'VGG16':
        model = getVGG16Architecture(classes, dropoutRate)
    elif modelArchitecture == 'VGG19':
        model = getVGG19Architecture(classes, dropoutRate)
    else:
        model = getInceptionV3Architecture(classes, dropoutRate)
    
    initialTraining(optimazerLastLayer, noOfEpochs, batchSize, savedModelName, train_generator, validation_generator, model)
    
    setLayersToRetrain(model, modelArchitecture)
    
    finetuningTraining(learningRate, noOfEpochs, batchSize, savedModelName, train_generator, validation_generator, model)

def main(args):
    pprint(args)
    if args.process_data:
        dataPreprocessing(args.car_ims_dir, args.car_ims_labels)
    
    K.clear_session()

    model(args.learning_rate, 
          args.optimizer_last_layer, 
          args.no_of_epochs, 
          args.batch_size, 
          args.saved_model_name,
          args.car_ims_dir, 
          args.car_ims_labels,
          args.model,
          args.dropout_rate)

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
        default=False)
        
    parser.add_argument('--car_ims_dir', type=str, 
        help='Directory where all pictures are located or where subfolder train/val are located', default='./car_ims')

    parser.add_argument('--car_ims_labels', type=str, 
        help='Points to the file with all labels', default='./cars_annos.mat')    

    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.0001)
    
    parser.add_argument('--dropout_rate', type=float,
        help='Fraction of the input units to drop.', default=0.5)
    
    parser.add_argument('--optimizer_last_layer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='RMSPROP')

    parser.add_argument('--model', type=str, choices=['VGG19', 'VGG16', 'InceptionV3'],
        help='The optimization algorithm to use', default='VGG16')

    parser.add_argument('--no_of_epochs', type=int,
        help='Number of epochs to run.', default=25)

    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=64)
    
    parser.add_argument('--saved_model_name', type=str,
        help='Number of images to process in a batch.', default='carRecognition')
    
    return parser.parse_args(argv)
    
if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
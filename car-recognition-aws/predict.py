'''
Created on 10 lut 2018

@author: mgdak
'''

import os
import argparse
import sys
from pprint import pprint
import shutil as sh
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop

def main(args):
    pprint(args)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        args.predict_data, 
        target_size=(224, 224), 
        batch_size=1, 
        class_mode='categorical')
    

    # load json and create model
    json_file = open(args.model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(args.model_name + '.h5')
    print('Loaded model from disk')
     
    # evaluate loaded model on test data
    sgd = SGD(lr=args.learning_rate, decay=args.lr_decay, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    score = loaded_model.predict_generator(test_generator, steps = args.no_of_samples, workers=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--predict_data', type=str,
        help='Dir with data to predict classes for'
        , default= './car_ims/tst')
        
    parser.add_argument('--model_name', type=str,
        help='Saved model name location', default='./vgg16/tst/vgg16_lr001_dr7_decaye-4_finalModel')
    
    parser.add_argument('--no_of_samples', type=int,
        help='Batch size', default=16)
        
    parser.add_argument('--lr_decay', type=float,
        help='Learning rate decay.', default=1e-4)
    
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.001)
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
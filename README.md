# Car classificator
This is a car classificator build using pretrained VGG16, VGG19 and InceptionV3 on ImageNet data set.

## Getting the data
The model has been trained on [Cars Dataset from Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Processing the data
Download the whole dataset (all images) and use following script to divide into train and test/validatoin `python matlabtester.py --process_data=True --car_ims_dir=DIR_WITH_ALL_IMAGES --car_ims_labels=cars_annos.mat`
If you need to create cross validation set, which I had to create but realized that later, use createTestFromValidation.py.
THe script will create cross validation dataset from your test set.

## Training the model
In order to train desired model use `python matlabtester.py --process_data=False --car_ims_dir=DIR_WITH_ALL_IMAGES --car_ims_labels=cars_annos.mat` and choose the model by providing the proper value in --mode selector.
On AWS p2.xlarge one steps takes ca. 2 sec. One epoch ca. 200 sec.

## The end result
Due to small size of the data set the simplest model turned out to be the most accurate.
We used early stopping to get rid of overfitting.
I managed to train VGG16 network with 66,11% accuracy on cross validation data set (drop out = 0.8, no learning rate decay). Below you will find the accuracy over epochs (red - val_accuracy, blue - train accuracy).
![alt text](https://github.com/michalgdak/car-recognition/blob/master/car-recognition-aws/VGG16_lr001_dr8_finalModel_plot.png "accuracy over epochs for VGG16")

The more complex model (ex. InceptionV3) the less accurate results are. This is understanable due to bias/variance problem.

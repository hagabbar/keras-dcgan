#!/usr/bin/env python
# Code to do parameter estimation using a generative adversarial 
# neural network.
# Author: Hunter Gabbard

import glob
import cPickle as pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import os,sys


class bbhparams:
    def __init__(self,mc,M,eta,m1,m2,ra,dec,iota,psi,idx,fmin,snr,SNR):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR

class network_args:
    def __init__(self, args):
        self.features = np.array(args.features.split(','))
        self.num_classes = 1
        self.class_weight = {0:args.noise_weight, 1:args.sig_weight}
        self.Nfilters = np.array(args.nfilters.split(",")).astype('int')
        self.kernel_size = np.array([i.split("-") for i in np.array(args.filter_size.split(","))]).astype('int')
        self.stride = np.array([i.split("-") for i in np.array(args.filter_stride.split(","))]).astype('int')
        self.dilation = np.array([i.split("-") for i in np.array(args.dilation.split(","))]).astype('int')
        self.activation = np.array(args.activation_functions.split(','))
        self.dropout = np.array(args.dropout.split(",")).astype('float')
        self.pooling = np.array(args.pooling.split(',')).astype('bool')
        self.pool_size = np.array([i.split("-") for i in np.array(args.pool_size.split(","))]).astype('int')
        self.pool_stride = np.array([i.split("-") for i in np.array(args.pool_stride.split(","))]).astype('int')

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model(shape):
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=shape)
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(args, netargs, shape, outdir, X_train, y_train, X_val, y_val, X_test, y_test, samp_weights=None):
    # define training/testing/validation sets here
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train = (X_train.astype(np.float32) - 127.5)/127.5
    #X_train = X_train[:, :, :, None]
    #X_test = X_test[:, :, :, None]

    

    # train models
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser(prog='dcgan.py', description='Generative Adversarial Neural Network in keras with tensorflow')
    parser.add_argument("--mode", type=str)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)

    # arguments for data
    parser.add_argument('-SNR', '--SNR', type=int,
                        help='')
    parser.add_argument('-trdt', '--training_dtype', type=str,
                        help='')
    parser.add_argument('-tsdt', '--testing_dtype', type=str,
                        help='')
    parser.add_argument('--datapath', type=str,
                        help='')
    parser.add_argument('-Nts', '--Ntimeseries', type=int, default=10000,
                        help='number of time series for training')
    #parser.add_argument('-ds', '--set-seed', type=str,
    #                    help='seed number for each training/validaiton/testing set')
    parser.add_argument('-Ntot', '--Ntotal', type=int, default=10,
                        help='number of available datasets with the same name as specified dataset')
    parser.add_argument('-Nval', '--Nvalidation', type=int, default=10000,
                        help='')

    # arguments for input data to network (e.g training/testing/validation data/params)
    parser.add_argument('-Trd', '--training_dataset', type=str,
                       default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav',
                       help='path to the data')
    parser.add_argument('-Trp', '--training_params', type=str, #nargs='+',
                       default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                       help='path to the training params')
    parser.add_argument('-Vald', '--validation_dataset', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav',
                        help='path to the data')
    parser.add_argument('-Valp', '--validation_params', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                        help='path to the validation params')
    parser.add_argument('-Tsd', '--test_dataset', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav',
                        help='path to the data')
    parser.add_argument('-Tsp', '--test_params', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                        help='path to the testing params')

    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='size of batches used for training/validation')
    parser.add_argument('-nw', '--noise_weight', type=float, default=1.0,
                        help='')
    parser.add_argument('-sw', '--sig_weight', type=float, default=1.0,
                        help='')
    # arguments for optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='SGD',
                        help='')
    parser.add_argument('-lr', '--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('-mlr', '--max_learning_rate', type=float, default=0.01,
                        help='max learning rate for cyclical learning rates')
    parser.add_argument('-NE', '--n_epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('-dy', '--decay', type=float ,default=0.0,
                        help='help')
    parser.add_argument('-ss', '--stepsize', type=float, default=500,
                        help='help')
    parser.add_argument('-mn', '--momentum', type=float, default=0.9,
                        help='momentum for updates where applicable')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='')
    parser.add_argument('--epsilon', type=float, default=1e-08,
                        help='')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='')
    parser.add_argument('-pt', '--patience', type=int, default=10,
                        help='')
    parser.add_argument('-lpt', '--LRpatience', type=int, default=5,
                        help='')

    #arguments for network
    parser.add_argument('-f', '--features', type=str, default="1,1,1,1,0,4" ,
                        help='order and types of layers to use, see RunCNN_bbh.sh for types')
    parser.add_argument('-nf', '--nfilters', type=str, default="16,32,64,128,32,2",
                        help='number of kernels/neurons per layer')
    parser.add_argument('-fs', '--filter_size', type=str, default="1-1-32,1-1-16,1-1-8,1-1-4,0-0-0,0-0-0" ,
                        help='size of convolutional layers')
    parser.add_argument('-fst', '--filter_stride', type=str, default="1-1-1,1-1-1,1-1-1,1-1-1",
                        help='stride for max-pooling layers')
    parser.add_argument('-fpd', '--filter_pad', type=str, default="0-0-0,0-0-0,0-0-0,0-0-0",
                        help='padding for convolutional layers')
    parser.add_argument('-dl', '--dilation', type=str, default="1-1-1,1-1-1,1-1-4,1-1-4,1-1-1",
                        help='dilation for convolutional layers, set to 1 for normal convolution')
    parser.add_argument('-p', '--pooling', type=str, default="1,1,1,1",
                        help='')
    parser.add_argument('-ps', '--pool_size', type=str, default="1-1-8,1-1-6,1-1-4,1-1-2",
                        help='size of max-pooling layers after convolutional layers')
    parser.add_argument('-pst', '--pool_stride', type=str, default="1-1-4,1-1-4,1-1-4,0-0-0,0-0-0",
                        help='stride for max-pooling layers')
    parser.add_argument('-ppd', '--pool_pad', type=str, default="0-0-0,0-0-0,0-0-0",
                        help='')
    parser.add_argument('-dp', '--dropout', type=str, default="0.0,0.0,0.0,0.0,0.1,0.0",
                        help='dropout for the fully connected layers')
    parser.add_argument('-fn', '--activation_functions', type=str, default='elu,elu,elu,elu,elu,softmax',
                        help='activation functions for layers')

    # general arguments
    parser.add_argument('-od', '--outdir', type=str, default='./history',
                        help='')
    parser.add_argument('--notes', type=str,
                        help='')

    args = parser.parse_args()
    return args

def concatenate_datasets(datapath, snr, training_dtype, testing_dtype, Nts, Nval = 10000, Ntot = 30):
    """
    shorten and concatenate data
    :param initial_dataset: first dataset in the set
    :param Nts: total number of images/time series
    :param Ntot: total number of available datasets
    :return:
    """

    print('Using data located in: {0}'.format(datapath))
    training_datasets = sorted(glob.glob('{0}/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, training_dtype)))
    validation_datasets = sorted(glob.glob('{0}/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, testing_dtype)))
    test_datasets = sorted(glob.glob('{0}/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, testing_dtype)))
    print(training_datasets, validation_datasets, test_datasets)


    print('Using data located in: {0}'.format(datapath))
    training_paramsets = sorted(glob.glob('{0}/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, training_dtype)))
    validation_paramsets = sorted(glob.glob('{0}/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, testing_dtype)))
    test_paramsets = sorted(glob.glob('{0}/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, testing_dtype)))
    print(training_paramsets, validation_paramsets, test_paramsets)

    # load in dataset 0 params and labels
    with open(training_datasets[0], 'rb') as rfp, open(training_paramsets[0], 'rb') as p:
        base_train_set = pickle.load(rfp)[0]
        base_train_par = np.array(pickle.load(p))
        base_train_set = [base_train_set, base_train_par]

        # set noise sample param values to zero
        # get desired parameter
        base_train_new = False
        for idx,i in enumerate(base_train_set[1]):
            if i != None and not base_train_new:
                base_train_new = [[base_train_set[0][idx]],[i.mc]]
            elif i != None and base_train_new:
                base_train_new[1].append(i.mc)
                base_train_new[0].append(base_train_set[0][idx])
        base_train_set = [np.array(base_train_new[0]),np.array(base_train_new[1])]  


    with open(validation_datasets[0], 'rb') as rfp, open(validation_paramsets[0], 'rb') as p:
        base_valid_set = pickle.load(rfp)[0]
        base_valid_par = np.array(pickle.load(p))
        base_valid_set = [base_valid_set, base_valid_par]

        # set noise sample param values to zero
        # get desired parameter
        base_valid_new = False
        for idx,i in enumerate(base_valid_set[1]):
            if i != None and not base_valid_new:
                base_valid_new = [[base_valid_set[0][idx]],[i.mc]]
            elif i != None and base_valid_new:
                base_valid_new[1].append(i.mc)
                base_valid_new[0].append(base_valid_set[0][idx])
        base_valid_set = [np.array(base_valid_new[0]),np.array(base_valid_new[1])]

    with open(test_datasets[0], 'rb') as rfp, open(test_paramsets[0], 'rb') as p:
        base_test_set = pickle.load(rfp)[0]
        base_test_par = np.array(pickle.load(p))
        base_test_set = [base_test_set, base_test_par]

        # set noise sample param values to zero
        # get desired parameter
        base_test_new = False
        for idx,i in enumerate(base_test_set[1]):
            if i != None and not base_test_new:
                base_test_new = [[base_test_set[0][idx]],[i.mc]]
            elif i != None and base_test_new:
                base_test_new[1].append(i.mc)
                base_test_new[0].append(base_test_set[0][idx])
        base_test_set = [np.array(base_test_new[0]),np.array(base_test_new[1])]


    # size of data sets
    size = int(1e4)
    val_size = int(1e4)
    # number of datasets -  depends on Nts
    Nds = np.floor(Nts / float(size))
    print(Nds)
    # check there are sufficient datasets
    if not Nds <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    # start with training set
    # if more than the initial data set is needed
    if Nds > 1:
        # how many images/time series needed
        need = Nts - size

        # loop over enough files to reach total number of time series
        for ps_idx,ds in enumerate(training_datasets[1:int(Nds)]):
            print(ds)
            with open(ds, 'rb') as rfp, open(training_paramsets[ps_idx], 'rb') as p:
                train_set = pickle.load(rfp)[0]
                train_par = np.array(pickle.load(p))
                train_set = [train_set, train_par]


                # set noise sample param values to zero
                # get desired parameter
                train_new = False
                for idx,i in enumerate(train_set[1]):
                    if i != None and not train_new:
                        train_new = [[train_set[0][idx]],[i.mc]]
                    elif i != None and train_new:
                        train_new[1].append(i.mc)
                        train_new[0].append(train_set[0][idx])
                train_set = [np.array(train_new[0]),np.array(train_new[1])]

            # check if this set needs truncating
            if need > size:
                cut = size
            else:
                cut = need

            # empty arrays to populate
            aug_train_set = np.zeros(2, dtype = np.ndarray) # change to two if wanting labels as well
            # concatenate the arrays
            for i in range(2): # change to 2 if also wanting labels
                aug_train_set[i] = np.concatenate((base_train_set[i], train_set[i][:cut]), axis=0)
            # copy as base set for next loop
            base_train_set = aug_train_set


            need -= size


    else:
        # return truncated version of the initial data set
        # change both 1 numbers back to 2 if wanting labels as well
        aug_train_set = np.zeros(2, dtype=np.ndarray)

        for i in range(2):
            aug_train_set[i] = base_train_set[i][:Nts]

        base_train_set = aug_train_set

    # validation/testing fixed at 10K
    Nds_val = np.floor(Nval / float(val_size))
    # check there are sufficient datasets
    if not Nds_val <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    if Nds_val > 1:
        # how many images/time series needed
        need = Nval - val_size


        # loop over enough files to reach total number of time series
        for Vds, Vps, Tds, Tps in zip(validation_datasets[1:int(Nds_val)], validation_paramsets[1:int(Nds_val)], test_datasets[1:int(Nds_val)], test_paramsets[1:int(Nds_val)]):
            with open(Vds, 'rb') as rfp, open(Vps, 'rb') as p:
                valid_set = pickle.load(rfp)[0]
                valid_params = np.array(pickle.load(p))
                valid_set = [valid_set,valid_params]

                # set noise sample param values to zero
                # get desired parameter
                valid_new = False
                for idx,i in enumerate(valid_set[1]):
                    if i != None and not valid_new:
                        valid_new = [[valid_set[0][idx]],[i.mc]]
                    elif i != None and valid_new:
                        valid_new[1].append(i.mc)
                        valid_new[0].append(valid_set[0][idx])
                valid_set = [np.array(valid_new[0]),np.array(valid_new[1])]

            with open(Tds, 'rb') as rfp, open(Tps, 'rb') as p:
                test_set = pickle.load(rfp)[0]
                test_params = np.array(pickle.load(p))
                test_set = [test_set,test_params]

                # set noise sample param values to zero
                # get desired parameter
                test_new = False
                for idx,i in enumerate(test_set[1]):
                    if i != None and not test_new:
                        test_new = [[test_set[0][idx]],[i.mc]]
                    elif i != None and test_new:
                        test_new[1].append(i.mc)
                        test_new[0].append(test_set[0][idx])
                test_set = [np.array(test_new[0]),np.array(test_new[1])]

            # check if this set needs truncating
            if need > val_size:
                cut = val_size
            else:
                cut = need

            # empty arrays to populate
            aug_valid_set = np.zeros(2, dtype = np.ndarray) # change back to 2 for labels
            aug_test_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels
            # concatenate the arrays
            for i in range(2): # change back to 2 for labels
                aug_valid_set[i] = np.concatenate((base_valid_set[i], valid_set[i][:cut]), axis=0)
                aug_test_set[i] = np.concatenate((base_test_set[i], test_set[i][:cut]), axis=0)

            # copy as base set for next loop
            base_valid_set = aug_valid_set
            base_test_set = aug_test_set

            need -= val_size


    else:
        # return truncated version of the initial data set
        aug_valid_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels
        aug_test_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels

        for i in range(2): # change back to 2 for labels
            aug_valid_set[i] = base_valid_set[i][:Nval]
            aug_test_set[i] = base_test_set[i][:Nval]

        base_valid_set = aug_valid_set
        base_test_set = aug_test_set

    return base_train_set, base_valid_set, base_test_set

def truncate_dataset(dataset, start, length):
    """
    :param dataset:
    :param start:
    :param end:
    :return:
    """
    print('    length of data prior to truncating: {0}'.format(dataset[0].shape))
    print('    truncating data between {0} and {1}'.format(start, start+length))
    # shape of truncated dataset
    new_shape = (dataset[0].shape[0],1,length)
    # array to populate
    #truncated_data = np.empty(new_shape, dtype=np.ndarray)
    # loop over data and truncate
    #for i,ts in enumerate(dataset[0]):
    #    truncated_data[i] = ts[0,start:(start+length)].reshape(1,length)

    dataset[0] = dataset[0][:,:,start:(start+length)]
    print('    length of truncated data: {}'.format(dataset[0].shape))
    return dataset

def load_data(args, netargs):
    """
    Load the data set
    :param dataset: the path to the data set (string)
    :param Nts: total number of time series for training
    :return: tuple of theano data set
    """

    train_set, valid_set, test_set = concatenate_datasets(args.datapath, args.SNR, args.training_dtype,
                                                          args.testing_dtype, args.Ntimeseries, args.Nvalidation, args.Ntotal)


    start = 4096
    length = 8192
    print('Truncating training set')
    train_set = truncate_dataset(train_set,start, length)
    print('Truncating validation set')
    valid_set = truncate_dataset(valid_set,start, length)
    print('Truncating test set')
    test_set = truncate_dataset(test_set, start, length)

    Ntrain = train_set[0].shape[0]
    xshape = train_set[0].shape[1]
    yshape = train_set[0].shape[2]
    channels = 1

    rescale = False

    if rescale:
        print('Rescaling data')
        for i in range(Ntrain):
            train_set[0][i] = preprocessing.normalize(train_set[0][i])

        for i in range(args.Nvalidation):
            valid_set[0][i] = preprocessing.normalize(valid_set[0][i])
            test_set[0][i] = preprocessing.normalize(test_set[0][i])

    x_train = (train_set[0].reshape(Ntrain, channels, xshape, yshape))
    #y_train = to_categorical(train_set[1], num_classes=netargs.num_classes)
    y_train = train_set[1]
    x_val = (valid_set[0].reshape(valid_set[0].shape[0], channels, xshape, yshape))
    #y_val = to_categorical(valid_set[1], num_classes=netargs.num_classes)
    y_val = valid_set[1]
    x_test = (test_set[0].reshape(test_set[0].shape[0], channels, xshape, yshape))
    #y_test = to_categorical(test_set[1], num_classes=netargs.num_classes)
    y_test = test_set[1]

    print('Traning set dimensions: {0}'.format(x_train.shape))
    print('Validation set dimensions: {0}'.format(x_val.shape))
    print('Test set dimensions: {0}'.format(x_test.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    args = get_args()
    # convert args to correct format for network
    netargs = network_args(args)

    # load in time series info
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args, netargs)

    if not os.path.exists('{0}/SNR{1}'.format(args.outdir,args.SNR)):
        os.makedirs('{0}/SNR{1}'.format(args.outdir,args.SNR))

    Nrun = 0
    while os.path.exists('{0}/SNR{1}/run{2}'.format(args.outdir,args.SNR,Nrun)):
        Nrun += 1
    os.makedirs('{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR, Nrun))

    shape = x_train.shape[1:]
    out = '{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR,Nrun)

    # train and test network
    if args.mode == "train":
        train(args, netargs, shape, out,
              x_train, y_train, x_val, y_val, x_test, y_test) # add final_tr_params at end if weighitng on params
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)

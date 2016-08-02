import midi
import numpy as np
np.random.seed(0)
import pygame
import test_play as play
import m2m
import os
import pickle
import sys
import target_generator as target
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import seaborn
import slapadabass as slap

def master_matrix(rootdir, divide_features=False):
    #Run through song folder and build matrix with all bass tracks
    if divide_features == False:
        master_matrix_list = []
        for subdir, dirs, files in os.walk(rootdir):
            for f in files:
                midi_matrx = m2m.mid_to_matrix(f)
                master_matrix_list.append(midi_matrx)

        with open('midi_matrix', 'w') as f:
            pickle.dump(master_matrix_list, f)

    else:
        pitch_mtrx = []
        length_mtrx = []
        velocity_mtrx = []

        for subdir, dirs, files in os.walk(rootdir):
            for f in files:
                pitch_array, length_array, velocity_array = m2m.mid_to_matrix('../Code/', f, 64, divide_features=True)
                pitch_mtrx.append(pitch_array)
                length_mtrx.append(length_array)
                velocity_mtrx.append(velocity_array)

        with open('pitch_matrix', 'w') as f:
            pickle.dump(pitch_mtrx, f)
        with open('length_matrix', 'w') as f:
            pickle.dump(length_mtrx, f)
        with open('velocity_matrix', 'w') as f:
            pickle.dump(velocity_mtrx, f)

def normalize_master_mtrx(matrix, feature):
    '''Normalizes matrix from 0-1.
    Max Note: 67
    Min Note: 35
    Max Length: 521
    Min Length: 1
    Max Velocity: 127
    Min Velocity: 30
    '''
    mtrx = np.array(matrix)
    mtrx_min = np.min(mtrx[np.nonzero(mtrx)])
    mtrx_max = mtrx.max()
    mtrx_range = mtrx_max - mtrx_min
    mtrx1 = mtrx.astype(float)
    mtrx2 = mtrx1 - mtrx_min
    mtrx3 = mtrx2/mtrx_range
    mtrx3[mtrx3 < 0] = -1

    mtrx_max_min = (mtrx_max, mtrx_min)

    if feature == 'pitch':
        with open('pitch_max_min', 'w') as f:
            pickle.dump(mtrx_max_min, f)
    elif feature == 'length':
        with open('length_max_min', 'w') as f:
            pickle.dump(mtrx_max_min, f)
    elif feature == 'velocity':
        with open('velocity_max_min', 'w') as f:
            pickle.dump(mtrx_max_min, f)
    else:
        print "Feature needs to be specified for proper storage: 'pitch', 'length', or 'velocity'"

    return mtrx3.tolist()

def normalize_input(matrix, feature):
    if feature == 'pitch':
        mtrx_max, mtrx_min = pickle.load(open('pitch_max_min', 'rb'))
    elif feature == 'length':
        mtrx_max, mtrx_min = pickle.load(open('length_max_min', 'rb'))
    elif feature == 'velocity':
        mtrx_max, mtrx_min = pickle.load(open('velocity_max_min', 'rb'))
    else:
        print "Feature needs to be specified for proper storage: 'pitch', 'length', or 'velocity'"

    mtrx = np.array(matrix)
    mtrx_range = mtrx_max - mtrx_min
    mtrx1 = mtrx.astype(float)
    mtrx2 = mtrx1 - mtrx_min
    mtrx3 = mtrx2/mtrx_range
    mtrx3[mtrx3 < 0] = -1

    return mtrx3.tolist()

def denormalize_output(matrix, feature):
    if feature == 'pitch':
        mtrx_max, mtrx_min = pickle.load(open('pitch_max_min', 'rb'))
    elif feature == 'length':
        mtrx_max, mtrx_min = pickle.load(open('length_max_min', 'rb'))
    elif feature == 'velocity':
        mtrx_max, mtrx_min = pickle.load(open('velocity_max_min', 'rb'))
    else:
        print "Feature needs to be specified for proper storage: 'pitch', 'length', or 'velocity'"

    mtrx = np.array(matrix)
    mtrx_range = mtrx_max - mtrx_min
    mtrx1 = mtrx * mtrx_range
    mtrx2 = mtrx1 + mtrx_min
    mtrx2[mtrx2 < mtrx_min] = 0

    return mtrx2.tolist()

def merge_all_to_mtrx(p_mtrx, l_mtrx, v_mtrx):
    merged = []
    for row in xrange(len(p_mtrx)):
        row_list = []
        for pos in xrange(len(p_mtrx[0])):
            pos_list = [p_mtrx[row][pos], l_mtrx[row][pos], v_mtrx[row][pos]]
            row_list.append(pos_list)
        merged.append(row_list)
    return merged


def build_data():
    rootdir = '/Users/claytonporter/Data_Science_Course/Project/Funk_Bass_Ready'
    master_matrix(rootdir, divide_features=True)



def grid_search(create_midi=True, validate=False):

    iterate = True
    training_model = 0
    grid_search_num = 11

    epoch_list = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
    RNN_list = [LSTM]
    optimizers = ['RMSprop']
    neurons = [128]

    save_version = 119
    input_basses = [1, 2, 3, 4, 5, 6]
    output_length = 128

    fig = plt.figure(figsize=(20, 20))
    i = 1
    for epochs in epoch_list:
        for arch in RNN_list:
            for opt in optimizers:
                for n in neurons:
                    if iterate == True:
                        # Load weights from specified model
                        loaded_model = load_prior_model(training_model=training_model)
                        loaded_model.compile(loss={'p_l_v': 'mean_squared_error'}, optimizer=opt)
                        history = fit_model(loaded_model, epochs)

                        #Save Model Weights
                        weights_fname = '../Models/model_weights{}.h5'.format(save_version)
                        loaded_model.save_weights(weights_fname, overwrite=True)

                        # Serialize Model architecture to JSON
                        model_json = loaded_model.to_json()
                        with open('../Models/model{}.json'.format(save_version), 'w') as json_file:
                            json_file.write(model_json)

                        training_model = save_version
                    else:
                        history = run_model(size=32, grid_search = True, cell=arch, neurons=n,\
                                            epochs=epochs, version=save_version, optimize_type=opt, plot=False)

                    train_losses = history.history['loss']

                    epoch_list = range(1, len(train_losses)+1)

                    if validate == True:
                        val_losses = history.history['val_loss']
                        min_val = round(min(val_losses), 4)
                        min_val_epoch = np.argmin(val_losses)+1
                    else:
                        pass

                    #Plotting
                    sub = fig.add_subplot(4,4,i)
                    sub.plot(epoch_list, train_losses, marker='.', label = 'Train Loss')
                    if validate == True:
                        sub.plot(epoch_list, val_losses, marker='.', label = 'Val Loss')
                    else:
                        pass

                    plt.legend(loc = 'upper right')

                    i += 1

                    if arch == GRU:
                        arch_name = 'GRU'
                    else:
                        arch_name = 'LSTM'

                    sub.set_title('{}, {}, and {} neurons'.format(arch_name, opt, n))
                    plt.xlabel('Epochs')
                    plt.ylabel('Mean Squared Error')
                    plt.ylim(0, 1)

                    if create_midi == True:
                        for bass in input_basses:
                            midi_file = 'test_input{}.mid'.format(bass)
                            raw_bass = slap.slapadabass(midi_file, 32, output_length, save_version)
                            slap.convert_output(raw_bass, save_version, bass)

                    save_version += 1

                    print 'Arch: ', arch
                    print 'Optimizer: ', opt
                    print 'Neurons: ', n
                    #print 'Min Val Loss: ', min_val
                    #print 'Min Val Epoch: ', min_val_epoch

    plt.savefig('../Grid_Search/grid_plot{}.png'.format(grid_search_num))
    plt.show()

def build_RNN(input_length, grid_search=False, cell=None, neurons=None, optimize_type=None):
    print 'Building Model...'
    bars = Input(shape=(input_length, 3), dtype='float32')

    if grid_search == True:
        layer = cell(neurons, return_sequences=False)(bars)

    else:
        layer = LSTM(128, return_sequences=True)(bars)
        #layer = GRU(2, return_sequences=True)(layer)
        layer = LSTM(128, return_sequences=False)(layer)
        #layer = LSTM(64, return_sequences=False)(layer)

    p_l_v = Dense(3, activation='linear', name='p_l_v')(layer)
    model = Model(input=bars, output=p_l_v)
    model.compile(loss={'p_l_v': 'mean_squared_error'}, optimizer=optimize_type)

    return model


def fit_model(model, num_epoch, validate=False):
    print 'Prepping Model...'
    pitch_matrix = pickle.load(open('pitch_matrix', 'rb'))
    length_matrix = pickle.load(open('length_matrix', 'rb'))
    velocity_matrix = pickle.load(open('velocity_matrix', 'rb'))

    #Normalize Matrices from 0-1 seperately
    p_norm = normalize_master_mtrx(pitch_matrix, 'pitch')
    l_norm = normalize_master_mtrx(length_matrix, 'length')
    v_norm = normalize_master_mtrx(velocity_matrix, 'velocity')

    #Merges normalized array back into one instance e.g [p, l, v] for each time step
    merged_norm = merge_all_to_mtrx(p_norm, l_norm, v_norm)

    #Splits all 208 bass lines into X (32 1/16th note) and a y (the 33rd 1/16th note)
    X, y = target.X_y_split(32, merged_norm)

    # Create validation X and y
    if validate == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_test = X[:641]
        y_test = y[:641]
        X_train = X[641:]
        y_train = y[641:]
        validation_data=(X_test, y_test)
        model.fit(X, y, batch_size=32, nb_epoch=num_epoch, validation_data=(X_test, y_test))
    # Fit model
    else:
        model.fit(X, y, batch_size=32, nb_epoch=num_epoch)

    history = model.history
    return history


def run_model(size=32, grid_search = False, cell=None, neurons=1, epochs=10, version=111,  optimize_type='RMSprop', plot=True):
    '''
    Builds and fits data to model, saving weights and architecture.

    INPUTS
    - size (int): Number of time steps in the input (typically 32)
    - epochs (int): Number of epochs you wish to run
    - version (int): version number to indicate when saving model to a file name

    OUTPUTS
    Pickles model weights and model architecture
    '''


    #build_RNN(input_length, grid_search=False, cell=None, =None, optimize_type=None
    model = build_RNN(size, grid_search=grid_search, cell=cell, neurons=neurons, optimize_type=optimize_type)
    history = fit_model(model, epochs)
    weights_fname = '../Models/model_weights{}.h5'.format(version)
    model.save_weights(weights_fname, overwrite=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open('../Models/model{}.json'.format(version), 'w') as json_file:
        json_file.write(model_json)

    if plot == True:
        train_losses = history.history['loss']
        val_losses = history.history['val_loss']
        epoch_list = range(1, len(train_losses)+1)

        min_val = round(min(val_losses), 4)
        min_val_epoch = argmin(val_losses)

        fig = plt.figure(figsize=(9, 6))
        plt.plot(epoch_list, train_losses, marker='.', label = 'Train Loss')
        plt.plot(epoch_list, val_losses, marker='.', label = 'Val Loss')
        plt.legend(loc = 'upper right')

        plt.title('Model #{}, 1 LSTM, {}, 16 neurons '.format(version, optimize_type))
        fig.text(0.1, 0.01, 'Val Loss Min: {} @ epoch: {}'.format(min_val, min_val_epoch),
                verticalalignment='bottom', horizontalalignment='center', fontsize=10)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.ylim(0, 0.6)

        plt.savefig('../Model_Plots/model{}_1LSTM_{}_16n.png'.format(version, optimize_type))
        plt.show()

    else:
        pass

    history = model.history
    return history


def load_prior_model(training_model=0):

    # load json and create model
    json_file = open('../Models/model{}.json'.format(training_model), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights('../Models/model_weights{}.h5'.format(training_model))
    print('Loaded model from disk')

    return loaded_model

def load_arch(training_model=0):
    '''Get summary of specific model architecture'''

    json_file = open('../Models/model{}.json'.format(training_model), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.summary()


if __name__ == '__main__':

    load_arch(training_model=114)

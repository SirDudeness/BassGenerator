import midi
import numpy as np
import pygame
import test_play as play
import m2m
import os
import pickle
import sys
import model_build
from keras.models import model_from_json

def slapadabass(midi_file, num_steps, output_length, version, preround=False):
    '''This function will generate a midi_file to be played'''

    #File path where you want to pull your initiation bass lines
    start_fpath = 'Start_Lines'

    #Get bass line into matrix form and seperate to normalize each value type
    pitch_mtrx, length_mtrx, velocity_mtrx = m2m.mid_to_matrix(start_fpath, midi_file, num_steps, divide_features=True)

    #Normalize (List)
    p_norm = model_build.normalize_input(pitch_mtrx, 'pitch')
    l_norm = model_build.normalize_input(length_mtrx, 'length')
    v_norm = model_build.normalize_input(velocity_mtrx, 'velocity')

    # Merge normalized pitch, length, velocity back for the prediction model
    merged_norm = m2m.feature_combine(p_norm, l_norm, v_norm)

    # Convert to numpy array
    X = np.array(merged_norm)
    X_exp = np.expand_dims(X, axis=0)

    # load json and create model
    json_file = open('../Models/model{}.json'.format(version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('../Models/model_weights{}.h5'.format(version))
    print('Loaded model from disk')

    whole_line = X.tolist()
    for i in xrange(output_length):

        next_step = loaded_model.predict(X_exp)
        if preround == True:
            #Split
            p_mtrx, l_mtrx, v_mtrx = m2m.feature_divide(next_step)

            #Denormalize each feature
            p_denorm = model_build.denormalize_output(p_mtrx, 'pitch')
            l_denorm = model_build.denormalize_output(l_mtrx, 'length')
            v_denorm = model_build.denormalize_output(v_mtrx, 'velocity')

            #Round denormalized output
            p_round = [float(round(p)) for p in p_denorm]
            l_round = [float(round(l)) for l in l_denorm]
            v_round = [float(round(v)) for v in v_denorm]

            #Re-Normalize
            p_norm = model_build.normalize_input(p_round, 'pitch')
            l_norm = model_build.normalize_input(l_round, 'length')
            v_norm = model_build.normalize_input(v_round, 'velocity')

            # Combine
            next_step = m2m.feature_combine(p_norm, l_norm, v_norm)
        else:
            pass
        X = np.reshape(X_exp, (32, 3))
        X = np.append(X, next_step, axis=0)
        next_step = np.reshape(next_step, (3, ))
        whole_line.append(next_step.tolist())
        X = np.delete(X, 0, 0)
        X_exp = np.expand_dims(X, axis=0)

    return whole_line

def convert_output(bass_line, version, bass_input):

    #Split
    p_mtrx, l_mtrx, v_mtrx = m2m.feature_divide(bass_line)
    #Denormalize each feature
    p_denorm = model_build.denormalize_output(p_mtrx, 'pitch')
    l_denorm = model_build.denormalize_output(l_mtrx, 'length')
    v_denorm = model_build.denormalize_output(v_mtrx, 'velocity')

    # Combine
    bass_matrix = m2m.feature_combine(p_denorm, l_denorm, v_denorm)

    #Convert to a midi file
    m2m.matrix_to_mid(bass_matrix, name='bass{}_model{}'.format(bass_input, version), BPM=55)

def gen_line(model_num=14, input_bass=1, output_length=128):
    # Generate bass line from specified model
    bass_line = slapadabass('test_input{}.mid'.format(input_bass), 32, output_length, model_num, preround=False)
    convert_output(bass_line, model_num, input_bass)

if __name__ == '__main__':

    gen_line(model_num=14, input_bass=1)

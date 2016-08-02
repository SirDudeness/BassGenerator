import pickle
import numpy as np

def X_y_split(split_number, matrix):
    '''SPLIT NUMBER = 32: This splits the bass line in 2 equal halves:
    The first 32 16ths determine the 33rd 16th note then moves down
    for a total of 32 X, y pairs per bass line.'''

    '''SPLIT NUMBER = 48: This splits the bass line in 3/4:
    The first 48 16ths determine the 49th 16th note then moves down
    for a total of 16 X, y pairs per bass line.'''
    X_list = []
    y_list = []
    for line in matrix:
        for i in xrange(64 - split_number):
            X = line[i:i + split_number]
            y = line[i + split_number]
            X_list.append(X)
            y_list.append(y)

    X_array = np.array(X_list)
    y_array = np.array(y_list)
    return X_array, y_array

if __name__ == '__main__':

    X, y = X_y_split(32, divide_features=True)

    print len(X)
    print len(y)

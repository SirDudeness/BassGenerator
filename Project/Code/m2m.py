import midi, numpy
import math
import os
import model_build as mb

def mid_to_matrix(folder, midi_file, num_steps, divide_features=False):
    '''Converts midi file into a matrix with a 3 dimentional entry for every
    16th note time step; e.g.

    [[pitch1, length1, velocity1],
    [pitch2, length2, velocity2],
    .
    .
    .
    [pitch64, length64, velocity64]]'''

    #read in midi file
    pattern = midi.read_midifile('../{0}/{1}'.format(folder, midi_file))

    #determine time unit
    note_event_list = []
    for track in pattern[0]:
        if isinstance(track, midi.NoteEvent):
            note_event_list.append(track)

    sixteenth_unit = 24

    #build matrix
    matrix = [[0, 0, 0] for _ in xrange(num_steps)]

    #If divide features
    pitch_mtrx = [0 for _ in xrange(num_steps)]
    length_mtrx = [0 for _ in xrange(num_steps)]
    velocity_mtrx = [0 for _ in xrange(num_steps)]

    time_running_total = 0
    for index, event in enumerate(note_event_list):
        time_running_total += event.tick

        #If note is an on event
        if isinstance(event, midi.NoteOnEvent) and event.velocity != 0:
            on_note = event.pitch
            rhythm_posn = time_running_total/sixteenth_unit
            i = index + 1
            i_length = 0

            #Find stop event
            unknown_note_length = True
            while unknown_note_length == True:
                if isinstance(note_event_list[i], midi.NoteOffEvent) and \
                note_event_list[i].pitch == on_note:
                    note_length = note_event_list[i].tick + i_length
                    unknown_note_length = False

                elif note_event_list[i].pitch == on_note and \
                note_event_list[i].velocity == 0:
                    note_length = note_event_list[i].tick + i_length
                    unknown_note_length = False

                else:
                    i_length += note_event_list[i].tick
                    i += 1

            #Append Note On Event to matrix
            if divide_features == False:
                matrix[rhythm_posn] = [on_note, note_length, event.velocity]
            else:
                pitch_mtrx[rhythm_posn] = on_note
                length_mtrx[rhythm_posn] = note_length
                velocity_mtrx[rhythm_posn] = event.velocity

        #Note Off Event
        else:
            pass

    if divide_features == False:
        return matrix
    else:
        return pitch_mtrx, length_mtrx, velocity_mtrx


def feature_combine(pitch_mtrx, length_mtrx, velocity_mtrx):
    '''Combines Pitch, Length, and Velocity arrays into one matrix with all
        three components'''
    combined_matrix = [[pitch_mtrx[i], length_mtrx[i], velocity_mtrx[i]] for i in xrange(len(pitch_mtrx))]
    return combined_matrix

def feature_divide(matrix):
    p_list = []
    l_list = []
    v_list = []
    for item in matrix:
        p_list.append(item[0])
        l_list.append(item[1])
        v_list.append(item[2])

    return p_list, l_list, v_list


def matrix_to_mid(matrix, name='example', BPM=55):
    '''Converts midi matrix back in a midi file so it can be played'''

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    tickscale = BPM
    note_scaler = float(BPM)/24
    running_tick = 0
    event_tick_total = 0
    for i in xrange(len(matrix)):
        note = int(matrix[i][0])
        length = int(math.ceil(matrix[i][1]*note_scaler))
        velocity = int(matrix[i][2])
        running_tick = i*tickscale
        if note == 0:
            pass

        else:
            if length == 0:
                length = 55
            else:
                pass
            on_tick = running_tick - event_tick_total
            on = midi.NoteOnEvent(tick=(on_tick), velocity=velocity, pitch=note)
            track.append(on)
            event_tick_total += on_tick
            #Check whether this is the last event
            try:
                last_event = matrix[i+1]
                #Find where to put stop event
                stop_event_steps = int(math.ceil(float(length)/tickscale))
                if stop_event_steps == 1:
                    if matrix[i+1][0] == 0:
                        off_tick = length
                    else:
                        off_tick = ((i + 1)*tickscale) - event_tick_total
                else:
                    for j in xrange(1, stop_event_steps):
                        if matrix[i+j][0] == 0:
                            off_tick = length
                        else:
                            off_tick = ((i + j)*tickscale) - event_tick_total
                            break
            except IndexError:
                off_tick = length

            #Add off event
            off = midi.NoteOffEvent(tick=off_tick, velocity=velocity, pitch=note)
            track.append(off)
            event_tick_total += off_tick

    eot = midi.EndOfTrackEvent(tick=tickscale*4)
    track.append(eot)

    try:
        midi.write_midifile("../Output/{}.mid".format(name), pattern)
    except:
        os.remove("../Output/{}.mid".format(name))
        print '{} failed to write to midi. Removing corrupted file...'.format(name)


if __name__ == '__main__':

    pitch_mtrx, length_mtrx, velocity_mtrx = mid_to_matrix('Funk_Bass_Ready','FantasticVoyage.mid', 64, divide_features=True)

    p_norm = mb.normalize_input(pitch_mtrx, 'pitch')
    l_norm = mb.normalize_input(length_mtrx, 'length')
    v_norm = mb.normalize_input(velocity_mtrx, 'velocity')

    funk_matrx = feature_combine(p_norm, l_norm, v_norm)

    print funk_matrx

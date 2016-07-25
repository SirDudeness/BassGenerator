import pygame

def play_music(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print "Music file %s loaded!" % music_file
    except pygame.error:
        print "File %s not found! (%s)" % (music_file, pygame.get_error())
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)



def playback(music_file):
    # pick a midi music file you have ...
    # (if not in working folder use full path)
    #music_file = "Funk_Bass_RAW/BoogieNights.mid"
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit


def test_track(filepath, output_name):
    pre_pattern = midi.read_midifile(filepath)
    print pre_pattern

    play.playback(filepath)

    test_matrx = m2m.mid_to_matrix(filepath)
    #print test_matrx
    m2m.matrix_to_mid(test_matrx, name=output_name)
    post_pattern = midi.read_midifile('{}.mid'.format(output_name))
    #print post_pattern
    play.playback('{}.mid'.format(output_name))
    #play.playback("../Funk_Bass_RAW/Voyager.mid")


if __name__ == '__main__':

    pass

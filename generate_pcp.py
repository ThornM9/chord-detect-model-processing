from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
import matplotlib.pyplot as plt
import numpy as np
import math; import statistics
import csv
CHORD_NAME = "a"
BASE_INPUT_FOLDER = "small_d/"
INPUT_FOLDER = BASE_INPUT_FOLDER + CHORD_NAME

chords = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
first = True
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

def write_csv(normalised_vectors):
    global first
    with open('second_chord_data.csv', mode='a', newline='') as chord_data_file:
        chord_writer = csv.writer(chord_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if first:
            # chord_writer.writerow(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'chord'])
            first = False
        for vec in normalised_vectors:
            vec.append(chords.index(CHORD_NAME))
            chord_writer.writerow(vec)


def main():
    mid_term_features, wav_file_list2, mid_feature_names = MidTermFeatures.directory_feature_extraction(INPUT_FOLDER, 1, 1, 0.3715192743764172, 0.3715192743764172)
    [sampling_rate, signal] = audioBasicIO.read_audio_file(wav_file_list2[0])
    print(sampling_rate)
    avg_vectors = []
    for elem in mid_term_features:
        avg_vectors.append(elem[21:33])
    print(mid_feature_names[21:33])

    normalised_vectors = []
    for vec in avg_vectors:
        total = sum(vec)
        new_vec = []
        for elem in vec:
            new_vec.append(elem / total)
        normalised_vectors.append(new_vec)
    
    #plot a normalised vector
    fig, ax = plt.subplots()
    xaxis = np.arange(12)
    plt.bar(xaxis, normalised_vectors[0])
    plt.xticks(xaxis, ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'))
    plt.show()

    write_csv(normalised_vectors)

# for i in range(len(chords)):
#     CHORD_NAME = chords[i]
#     INPUT_FOLDER = BASE_INPUT_FOLDER + CHORD_NAME
main()
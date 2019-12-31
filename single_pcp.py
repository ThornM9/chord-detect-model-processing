from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
import matplotlib.pyplot as plt
import numpy as np
import math; import statistics

INPUT_FILE = "predict/d8.wav"

[Fs, x] = audioBasicIO.read_audio_file(INPUT_FILE)
mid_term_features, mid_feature_names = MidTermFeatures.mid_feature_extraction(x, Fs, 1, 1, 0.05, 0.05)
print(mid_term_features)
avg_vector = []
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

print(normalised_vectors[0])

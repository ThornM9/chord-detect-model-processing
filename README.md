# ChordDetect Model Processing
This is the repository responsible for processing data and creating models for the chorddetect.com web application (source code: https://github.com/ThornM9/chord-detect-app).

This repository includes two tools. 
The first is a NodeJS command line tool in the process_wav directory which is used for processing a dataset of organised WAV files into a csv file that contains the Pitch Class Profile(PCP) of each file. 

The second tool is the train_model.py file which takes the training output csv file from process_wav and a validation output csv file and trains a Keras model which can be converted to a tensorflowJS model and used in the ChordDetect web application.

The neural network is based loosely on the structure defined in this research paper: https://www.researchgate.net/publication/252067543_Neural_networks_for_musical_chords_recognition.
A few changes were made to the structure that have reliably provided marginally better results. The neural network has a single dense hidden layer of 35 neurons and 10 output neurons. The loss function is a Keras defined Sparse Categorical Crossentropy function and the gradient function
is a Keras defined stochastic gradient descent algorithm with a learning rate of 0.01. This training method consistently scores approximately 99.6% on the validation dataset after 400 epochs for guitar chords.

Although this model was trained on guitar chords, the PCP for other instruments is quite similar, as shown in the research paper this process is based on. This means it can still predict the chord with high accuracy for other instruments.

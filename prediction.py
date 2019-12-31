import tensorflow as tf

MODEL = "guitar_chord_detection.h5" # filepath to trained model
model = tf.keras.models.load_model(MODEL)

predict_dataset = tf.convert_to_tensor([
    [[0.02899254903763187, 0.052036608157709455, 0.23884433932446367, 0.0052013471783651085, 0.14239997551388406, 0.07160815548480552, 0.12874564227984228, 0.010193506917486784, 0.008776093907768116, 0.1468352504841019, 0.15517748442721063, 0.011189047286730614],
    [0.015171499144598049, 0.005813379750523247, 0.3579707306476291, 0.011091468185118858, 0.062055285335763176, 0.02066099258042598, 0.3795782793758687, 0.0027860025616652795, 0.003462365971564795, 0.1052443841556951, 0.03427384159568347, 0.0018917706954642989]]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    chord_idx = tf.argmax(logits).numpy()
    print(chord_idx)
    p = tf.nn.softmax(logits)[chord_idx]
    name = chord_names[chord_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
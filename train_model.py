import tensorflow as tf
import matplotlib.pyplot as plt

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# define the csv format
column_names = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'chord']
feature_names = column_names[:-1]
label_name = column_names[-1]
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

# mapping of chord names to integer
chord_names = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']


# define the training dataset and read from the chord_data csv file
batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset("chord_data.csv", 
    batch_size, 
    column_names=column_names, 
    label_name=label_name, 
    num_epochs=1)

# map the training dataset to a more simplified data structure
train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

# define the structure of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(35, activation=tf.nn.relu, input_shape=(12,)), # first layer, requires the input_shape(number of input features)
    tf.keras.layers.Dense(10) # output neurons
])

# defining the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)

# defining the gradient used to optimize the model
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# set up the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# track the loss and accuracy over time
train_loss_results = []
train_accuracy_results = []

# train loop
num_epochs = 401
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # training loop
    for x, y in train_dataset:
        # optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # track progress
        epoch_loss_avg(loss_value) # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(y, model(x))
    
    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

# plot the training metrics
# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# fig.suptitle('Training Metrics')

# axes[0].set_ylabel("Loss", fontsize=14)
# axes[0].plot(train_loss_results)

# axes[1].set_ylabel("Accuracy", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)
# axes[1].plot(train_accuracy_results)
# plt.show()
# model.save("guitar_chord_detection.h5")

# define the validation/testing dataset
test_dataset = tf.data.experimental.make_csv_dataset(
    "chord_testing_data.csv",
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
    shuffle=False
)

test_dataset = test_dataset.map(pack_features_vector)

# test the accuracy of the trained model
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

predict_dataset = tf.convert_to_tensor([
    [0.02899254903763187, 0.052036608157709455, 0.23884433932446367, 0.0052013471783651085, 0.14239997551388406, 0.07160815548480552, 0.12874564227984228, 0.010193506917486784, 0.008776093907768116, 0.1468352504841019, 0.15517748442721063, 0.011189047286730614],
    [0.015171499144598049, 0.005813379750523247, 0.3579707306476291, 0.011091468185118858, 0.062055285335763176, 0.02066099258042598, 0.3795782793758687, 0.0027860025616652795, 0.003462365971564795, 0.1052443841556951, 0.03427384159568347, 0.0018917706954642989],
    [0.08039864945915239,0.08094136576090137,0.0717124755155601,0.08271199160431474,0.09857323647667877,0.07703149047876125,0.076636953643496,0.07764808613065968,0.08027450354739103,0.09218244132186403,0.09454068561865145,0.0873481204425693],
    [0.09601789166341317,0.09294073969704496,0.08065862346803307,0.07733956762278558,0.07510546943491546,0.07453829876921683,0.07916124597538231,0.07563696819713221,0.07943256171262586,0.08532849384604643,0.09103249531070629,0.09280764430269778],
    [0.08811265664964975,0.0827314859321838,0.08416886324118437,0.09553510095355651,0.0728597201637187,0.07715263458205683,0.0774314138922293,0.0776049963598588,0.08059103663955501,0.0885348382587718,0.08844119967585591,0.08683605365137924]
    
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    chord_idx = tf.argmax(logits, 1).numpy()
    print(chord_idx)
    p = tf.nn.softmax(logits)[chord_idx]
    name = chord_names[chord_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
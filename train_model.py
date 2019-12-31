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
train_dataset = tf.data.experimental.make_csv_dataset("process_wav/data.csv", 
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
model.save("model/chord_detection.h5")

# define the validation/testing dataset
test_dataset = tf.data.experimental.make_csv_dataset(
    "process_wav/validation_data.csv",
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
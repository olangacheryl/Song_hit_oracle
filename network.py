# import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
import os
# filter out any unwanted warnings from printing to console
import warnings
warnings.filterwarnings("ignore")

# join all the csv files
folder_path = '/Users/sabrinahatch/PycharmProjects/ANN_FInal/spotify_hits_data'
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)

# now, we clean the data
# let's start by dropping any categorical features
data.drop(["track","artist","uri"],axis=1,inplace=True)

# shuffle the data
data = data.sample(frac=1)

# now we preprocess the data
# extract the features
unscaled_inputs = data.iloc[:,0:-1]
# extract the hit/flop target
target = data.iloc[:,[-1]]
# scale the inputs
scaled_inputs = preprocessing.scale(unscaled_inputs)

# split our cleaned dataset into training, validation, and testing
# let's set up numpy arrays with the correct sizes for each to do this
# we will do 80% training, and then 10% for testing and validation respectively
total_data_count = scaled_inputs.shape[0]
training_data_count = int(0.8*total_data_count)
validation_data_count = int(0.1*total_data_count)
test_samples_count = total_data_count - training_data_count - validation_data_count

# now we populate the numpy arrays with data for each
training_inputs = scaled_inputs[:training_data_count]
training_targets = target[:training_data_count]
validation_inputs = scaled_inputs[training_data_count:training_data_count+validation_data_count]
validation_targets = target[training_data_count:training_data_count+validation_data_count]
test_inputs = scaled_inputs[training_data_count+validation_data_count:]
test_targets = target[training_data_count+validation_data_count:]

# save the three sets into .npz
np.savez('Spotify_training_data', inputs=training_inputs, targets=training_targets)
np.savez('Spotify_validation_data', inputs=validation_inputs, targets=validation_targets)
np.savez('Spotify_testing_data', inputs=test_inputs, targets=test_targets)

# create neural network

# start by loading data into npz type
npz = np.load('Spotify_training_data.npz')
# make sure they are all stored as floats
training_inputsn_inputs = npz['inputs'].astype(float)
npz = np.load('Spotify_validation_data.npz')
validation_inputs = npz['inputs'].astype(float)
validation_targets = npz['targets'].astype(int)
npz = np.load('Spotify_testing_data.npz')
test_inputs = npz['inputs'].astype(float)
test_targets = npz['targets'].astype(int)

# set the input and output sizes
input_size = 15 # count of features
output_size = 2 # count of targets
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

# define the optimizer and loss function
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# now we train our model
batch_size = 500
max_epochs = 20

# fit the model
history = model.fit(  training_inputs,
                      training_targets,
                      batch_size = batch_size,
                      epochs = max_epochs,
                      validation_data=(validation_inputs, validation_targets),
                      verbose = 2
          )

# visualize the ANN's loss
# get training and loss history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# now, we test the accuracy of the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
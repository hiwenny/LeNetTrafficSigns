
# Imports
import pickle
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
#% matplotlib inline

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_validation = len(X_validation)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 1

from tensorflow.contrib.layers import flatten
# Feed in X_input, image_shape, n_classes
def LeNet(x, input_shape, output_classes):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, input_shape[2], 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conn1 = flatten(conv2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    conn1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    conn1_b = tf.Variable(tf.zeros(120))
    conn1   = tf.matmul(conn1, conn1_W) + conn1_b
    
    # TODO: Activation.
    conn1 = tf.nn.relu(conn1)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    conn2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    conn2_b = tf.Variable(tf.zeros(84))
    conn2   = tf.matmul(conn1, conn2_W) + conn2_b
    
    # TODO: Activation.
    conn2 = tf.nn.relu(conn2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    conn3_W = tf.Variable(tf.truncated_normal(shape=(84, output_classes), mean = mu, stddev = sigma))
    conn3_b = tf.Variable(tf.zeros(output_classes))
    logits  = tf.matmul(conn2, conn3_W) + conn3_b
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x, image_shape, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# ==================================================== #

# NORMALIZED GRAY

def convert_to_grayscale(X_data):
    bat = []
    for i in range(0, len(X_data)):
        image = cv2.cvtColor(X_data[i], cv2.COLOR_RGB2GRAY)
        bat.append(image)
    X_data = np.reshape(bat, (-1, 32, 32, 1))
    return X_data

# Convert image to grayscale
X_train_gray = convert_to_grayscale(X_train)
X_test_gray = X_test
#X_test_gray = convert_to_grayscale(X_test)
#X_validation_gray = convert_to_grayscale(X_validation)
X_validation_gray = X_validation

X_train_gray, y_train = shuffle(X_train_gray, y_train)

# Normalization to center the image value distribution at 0.
# As recommended, image normalization is by (x - 128) / 128
X_train_normalized_gray = (X_train_gray - 128.0)/128.0
X_test_normalized_gray = X_test_gray
X_validation_normalized_gray = X_validation_gray
#X_test_normalized_gray = (X_test_gray - 128.0)/128.0
#X_validation_normalized_gray = (X_validation_gray - 128.0)/128.0

print(np.mean(X_train_normalized_gray))
print(np.mean(X_test_normalized_gray))
print(np.mean(X_validation_normalized_gray))
print('RGB norm shape:', X_train_normalized_gray.shape)
print('RGB norm shape indiv:', X_train_normalized_gray[0].shape)
print('y shape:', y_train.shape)

image_shape = X_train_normalized_gray[0].shape
x = tf.placeholder(tf.float32, (None, 32, 32, image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x, image_shape, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

N_EPOCHS = 10
index = random.randint(0, len(X_train_normalized_gray))
image = X_train_normalized_gray[index].squeeze()

#plt.figure(figsize=(1,1))
#plt.title(y_train[index])
#plt.imshow(image)

X_train_normalized_gray, y_train = shuffle(X_train_normalized_gray, y_train)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_normalized_gray)
    
    print("Training normalized gray...")
    print()
    for i in range(N_EPOCHS):
        X_train_normalized_gray, y_train = shuffle(X_train_normalized_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_normalized_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        validation_accuracy = evaluate(X_validation_normalized_gray, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, './lenetnormalizedgray')
    print("Model saved")

sess.close()

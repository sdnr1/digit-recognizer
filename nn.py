import numpy as np
from numpy import genfromtxt
import tensorflow as tf

DATA = genfromtxt('train.csv', delimiter=',')

train_dataset = DATA[1:, 1:].reshape((-1, 28, 28, 1)).astype(np.float32)
train_labels = DATA[1:, 0]

DATA = genfromtxt('test.csv', delimiter=',')

test_dataset = DATA[1:, :].reshape((-1, 28, 28, 1)).astype(np.float32)
del DATA

print("Train dataset\t:", train_dataset.shape)
print("Train labels\t:", train_labels.shape)
print("Test dataset\t:", test_dataset.shape)

# !!! zero centering and normalization of data is pending

train_labels = (np.arange(10) == train_labels[:, None]).astype(np.float32)
print(train_labels)
print()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

num_labels = 10
image_size = 28
epoches = 1
batch_size = 20
patch_size = 5
depth1 = 16
depth2 = 64
h0_layer_size = (image_size // 4) * (image_size // 4) * depth2
h1_layer_size = 512
h2_layer_size = 128
beta = 0.001

def compute_logits_with_dropout(dataset, weights, biases, keep_prob):
    
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dataset, weights[0], strides=[1, 1, 1, 1], padding='SAME'), biases[0]))

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001/9.0, beta=0.75)

    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, weights[1], strides=[1, 1, 1, 1], padding='SAME'), biases[1]))

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.nn.local_response_normalization(pool2, bias=1.0, alpha=0.001/9.0, beta=0.75)

    shape = pool2.get_shape().as_list()
    hidden = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

    layer1 = tf.nn.relu(tf.matmul(hidden, weights[2]) + biases[2])
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

    layer2 = tf.nn.relu(tf.matmul(layer1, weights[3]) + biases[3])
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

    logits = tf.matmul(layer2, weights[4]) + biases[4]
    return logits

def compute_logits(dataset, weights, biases):
    
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dataset, weights[0], strides=[1, 1, 1, 1], padding='SAME'), biases[0]))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001/9.0, beta=0.75)
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, weights[1], strides=[1, 1, 1, 1], padding='SAME'), biases[1]))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.nn.local_response_normalization(pool2, bias=1.0, alpha=0.001/9.0, beta=0.75)
    shape = pool2.get_shape().as_list()
    hidden = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    layer1 = tf.nn.relu(tf.matmul(hidden, weights[2]) + biases[2])
    layer2 = tf.nn.relu(tf.matmul(layer1, weights[3]) + biases[3])
    logits = tf.matmul(layer2, weights[4]) + biases[4]
    return logits

graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(train_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = [
        tf.Variable(tf.random_normal([patch_size, patch_size, 1, depth1], stddev=0.01, dtype=tf.float32)),
        tf.Variable(tf.random_normal([patch_size, patch_size, depth1, depth2], stddev=0.01, dtype=tf.float32)),
        tf.Variable(tf.random_normal([h0_layer_size, h1_layer_size], stddev=0.01, dtype=tf.float32)),
        tf.Variable(tf.random_normal([h1_layer_size, h2_layer_size], stddev=0.01, dtype=tf.float32)),
        tf.Variable(tf.random_normal([h2_layer_size, num_labels], stddev=0.01, dtype=tf.float32))
    ]

    biases = [
        tf.Variable(tf.zeros([depth1], dtype=tf.float32)),
        tf.Variable(tf.zeros([depth2], dtype=tf.float32)),
        tf.Variable(tf.zeros([h1_layer_size], dtype=tf.float32)),
        tf.Variable(tf.zeros([h2_layer_size], dtype=tf.float32)),
        tf.Variable(tf.zeros([num_labels], dtype=tf.float32))
    ]

    keep_prob = 0.75
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.02, global_step, 2100, 0.91, staircase=True)
    logits = compute_logits_with_dropout(tf_train_dataset, weights, biases, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta * (tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(biases[0]) + tf.nn.l2_loss(weights[1]) + tf.nn.l2_loss(weights[2]) + tf.nn.l2_loss(weights[3]) + tf.nn.l2_loss(weights[4]))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(compute_logits(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(compute_logits(tf_test_dataset, weights, biases))


final_predictions = np.zeros((28000, 10))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for epoch in range(epoches):
        print("\nEpoch : %d\n" % (epoch+1))
        offset = 0
        while offset < train_dataset.shape[0] :

            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size)]
            offset += batch_size

            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

            if(offset % 4200 == 0):
                print("Minibatch loss at %06d: %f\tAccuracy: %.1f%%" % (offset, l, accuracy(predictions, batch_labels)))
        
        # print("\nValidation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), train_labels))
    final_predictions = test_prediction.eval()

preds = np.argmax(final_predictions, 1)
preds = np.resize(preds, (28000, 1))
op = np.arange(1, 28001)
op = np.resize(op, (28000, 1))
op = np.hstack((op, preds))

with open("predictions.csv", "wb") as f:
    f.write(b'ImageId,Label\n')
    np.savetxt(f, op, fmt='%i', delimiter=",")
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
number_of_inputs = 28 * 28
n_hidden1 = 1000
n_hidden2 = 500
n_hidden3 = 100
n_hidden4 = 50
n_output = 10

def neuron_layer(X, n_neurons, name):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        # truncated normal distributions limit the size of the weights, speeding up the training time.
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        return tf.nn.relu(z)

x = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name="input")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def create_graph(x, y):
    with tf.name_scope("dnn"):
        hidden1_output = neuron_layer(x, n_hidden1, "hidden1")
        hidden2_output = neuron_layer(hidden1_output, n_hidden2, "hidden2")
        hidden3_output = neuron_layer(hidden2_output, n_hidden3, "hidden3")
        hidden4_output = neuron_layer(hidden3_output, n_hidden4, "hidden4")
        logits = neuron_layer(hidden4_output, n_output, "output")

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name="loss")

    with tf.name_scope("training"):
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        k = 1
        correctness = tf.nn.in_top_k(logits, y, k)
        accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32)) * 100
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    interim_checkpoint_path = "./checkpoints/mnist_model.ckpt"

    from datetime import datetime

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    log_dir = "{}/run-{}/".format(root_logdir, now)

    loss_summary = tf.summary.scalar('loss', loss)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    return logits

epochs = 10
batch_size = 50
n_batches = int(np.ceil(mnist.train.num_examples // batch_size))

early_stopping_check_frequency = n_batches // 10
early_stopping_check_limit = n_batches * 2

def accuracy(predictions, labels):
    correctness = np.argmax(predictions, 1) == np.argmax(labels, 1)
    return (100.0 * np.average(correctness, axis=0))
    #return (100.0 * np.sum(np.argmax(predictions, 0) == np.argmax(labels, 0))
    #    / predictions.shape[0])

with tf.Session("grpc://localhost:2222") as sess:
    # Don't initialise the variables as we will use the existing resource containers populated by other scripts.
    with tf.device("/job:local/task:0"):
        with tf.container("one"):
            logits_0 = create_graph(x, y)
            softmax_0 = tf.nn.softmax(logits_0)
            #output_0 = tf.argmax(softmax_0, axis=1)
    
    with tf.device("/job:local/task:1"):
        with tf.container("two"):
            logits_1 = create_graph(x, y)
            softmax_1 = tf.nn.softmax(logits_1)
            #output_1 = tf.argmax(softmax_1, axis=1)
    
    with tf.device("/job:local/task:2"):
        with tf.container("three"):
            logits_2 = create_graph(x, y)
            softmax_2 = tf.nn.softmax(logits_2)
            #output_2 = tf.argmax(softmax_2, axis=1)

    print("softmax_0.shape", softmax_0.shape)
    #print("output_0.shape", output_0.shape)
    # Merge the outputs into one tensor
    #outputs = tf.concat([output_0, output_1, output_2], axis=0)
    softmaxes = tf.stack([softmax_0, softmax_1, softmax_2])
    print("softmaxes.shape", softmaxes.shape)
    #print("outputs.shape", outputs.shape)
    # Find the average of the outputs, then pick the biggest, which is essentially the most voted output.
    ensemble_output = tf.reduce_mean(softmaxes, axis=0)
    print("ensemble_output.shape", ensemble_output.shape)
    eo = sess.run(ensemble_output, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    validation_acc = accuracy(eo, y)
    #print("ensemble_accuracy.shape", ensemble_accuracy.shape)

    #validation_acc = sess.run(ensemble_accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    print("validation accuracy", validation_acc)

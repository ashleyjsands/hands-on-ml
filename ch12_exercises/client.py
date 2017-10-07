import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser(description="a tensorflow MNIST client.")
parser.add_argument('server_host', type=str, help='the server host name and port')
parser.add_argument('container_name', type=str, help='the name of the resource container to use', nargs='?', default='')
parser.add_argument('learning_rate', type=float, help='the learning rate of the optimizer', nargs='?', default=0.1)
parser.add_argument('batch_size', type=int, help='the size of the batches when applying batch gradient descent', nargs='?', default=100)
args = parser.parse_args()

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

with tf.container(args.container_name):
    x = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name="input")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    # This is currently used.
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

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
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
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

epochs = 10
batch_size = args.batch_size
n_batches = int(np.ceil(mnist.train.num_examples // batch_size))

early_stopping_check_frequency = n_batches // 10
early_stopping_check_limit = n_batches * 2

print("Connecting to %s to use '%s' container" % (args.server_host, args.container_name))
print("Training with a learning rate of %s and a batch size of %s" % (args.learning_rate, args.batch_size))
with tf.Session("grpc://" + args.server_host) as sess:
    sess.run(init)
    best_validation_acc = 0.0
    best_validation_step = 0
    for epoch in range(epochs):
        print("epoch", epoch)
        for batch_index in range(n_batches):
            step = epoch * n_batches + batch_index
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            #if batch_index % 10 == 0:
                # Output summaries
                #summary_str = summary_op.eval(feed_dict={x: X_batch, y: y_batch, is_training: False})
                #file_writer.add_summary(summary_str, step)
            t, l, a = sess.run([training_op, loss, accuracy], feed_dict={x: X_batch, y: y_batch, is_training: True})
            if batch_index % 10 == 0: print("loss:", l, "training accuracy:", a)
            # Early stopping check
            if batch_index % early_stopping_check_frequency == 0:
                validation_acc = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels, is_training: False})
                print("validation accuracy", validation_acc)
                if validation_acc > best_validation_acc:
                    #print("Saving best model")
                    #saver.save(sess, early_stopping_checkpoint_path)
                    best_validation_acc = validation_acc
                    best_validation_step = step
                elif step >= (best_validation_step + early_stopping_check_limit):
                    print("Stopping early during epoch", epoch)
                    print("Best validation performance", best_validation_acc)
                    break
        else:
            continue
        break
        #save_path = saver.save(sess, interim_checkpoint_path)
    #saver.restore(sess, early_stopping_checkpoint_path)
    #test_acc = accuracy.eval(feed_dict={x: test_zero_to_four_images, y: test_zero_to_four_labels, is_training: False})
    #print(">>>>>>>>>> test dataset accuracy:", test_acc)

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="a tensorflow MNIST client.")
parser.add_argument('server_host', type=str, help='the server host name and port')
args = parser.parse_args()

c = tf.constant("hello world")

with tf.Session("grpc://" + args.server_host) as sess:
	print(c.eval())

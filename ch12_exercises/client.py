import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="a tensorflow MNIST client.")
parser.add_argument('server_host', type=str, help='the server host name and port')
parser.add_argument('container_name', type=str, help='the name of the resource container to use', nargs='?', default='')
args = parser.parse_args()

with tf.container(args.container_name):
	c = tf.constant("hello world")

print("Connecting to %s to use '%s' container" % (args.server_host, args.container_name))
with tf.Session("grpc://" + args.server_host) as sess:
	print(c.eval())

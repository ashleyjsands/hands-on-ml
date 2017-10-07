import tensorflow as tf

c = tf.constant("hello world")

with tf.Session("grpc://localhost:2222") as sess:
	print(c.eval())

with tf.Session("grpc://localhost:2223") as sess:
	print(c.eval())

with tf.Session("grpc://localhost:2224") as sess:
	print(c.eval())

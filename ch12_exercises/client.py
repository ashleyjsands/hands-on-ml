import tensorflow as tf

c = tf.constant("hello world")

# Tell each server to use only (almost) a third of the GPUs memory so they can all fit in the same GPU. 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33

with tf.Session("grpc://localhost:2222", config=config) as sess:
	print(c.eval())

with tf.Session("grpc://localhost:2223", config=config) as sess:
	print(c.eval())

with tf.Session("grpc://localhost:2224", config=config) as sess:
	print(c.eval())

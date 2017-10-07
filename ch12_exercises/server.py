import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="a tensorflow server.")
parser.add_argument('task_index', metavar='i', type=int, help='the task index for the server within the cluster')
args = parser.parse_args()

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"]})
# Tell each server to use only (almost) a third of the GPUs memory so they can all fit in the same GPU. 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=False)
config = tf.ConfigProto(gpu_options=gpu_options)
server = tf.train.Server(cluster, config=config, job_name="local", task_index=args.task_index)
server.join()

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="a tensorflow server.")
parser.add_argument('task_index', metavar='i', type=int, help='the task index for the server within the cluster')
args = parser.parse_args()

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"]})
server = tf.train.Server(cluster, job_name="local", task_index=args.task_index)
server.join()

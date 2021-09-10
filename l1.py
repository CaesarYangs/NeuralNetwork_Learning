import tensorflow as tf

'''
with tf.device("cpu"):
    a = tf.constant([1])



with tf.device("gpu"):
    b = tf.range(4)
print(a.device)
print(b.device)'''

from tensorflow.python.client import device_lib

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)
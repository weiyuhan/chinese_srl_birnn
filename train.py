import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF

#args
parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="the path of the train file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("val_path", help="the path of the validation file")
parser.add_argument("-e","--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_path = args.train_path
save_path = args.save_path
val_path = args.val_path
num_epochs = args.epoch
gpu_config = "/gpu:"+str(args.gpu)
# gpu_config = "/cpu:0"
num_steps = 200 # it must consist with the test

#get trainData and devData
start_time = time.time()
print "preparing train and validation data"
train_data, val_data = helper.getTrain(train_path=train_path, val_path=val_path, seq_max_len=num_steps)

#feature for trainData
X_train = train_data['char']
X_left_train = train_data['left']
X_right_train = train_data['right']
X_pos_train = train_data['pos']
X_lpos_train = train_data['lpos']
X_rpos_train = train_data['rpos']
X_rel_train = train_data['rel']
X_dis_train = train_data['dis']
y_train = train_data['label']

#feature for devData
X_val = val_data['char']
X_left_val = val_data['left']
X_right_val = val_data['right']
X_pos_val = val_data['pos']
X_lpos_val = val_data['lpos']
X_rpos_val = val_data['rpos']
X_rel_val = val_data['rel']
X_dis_val = val_data['dis']
y_val = val_data['label']

#dictionary
char2id, id2char = helper.loadMap("char2id")
pos2id, id2pos = helper.loadMap("pos2id")
label2id, id2label = helper.loadMap("label2id")

num_chars = len(id2char.keys())
num_poses = len(id2pos.keys())
num_classes = len(id2label.keys())
num_dises = 250

embedding_matrix = None
print "building model"
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	with tf.device(gpu_config):
		#initail Model
		initializer = tf.random_uniform_initializer(-0.1, 0.1)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = BILSTM_CRF(num_chars=num_chars, num_poses=num_poses, num_dises=num_dises, num_classes=num_classes, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, is_training=True)

		#train
		print "training model"
		tf.global_variables_initializer().run()
		model.train(sess, save_path, train_data, val_data)

		print "final best f1 is: %f" % (model.max_f1)

		end_time = time.time()
		print "time used %f(hour)" % ((end_time - start_time) / 3600)
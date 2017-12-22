import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import copy, os
from BILSTM_CRF_tf import BILSTM_CRF

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="the path of model file")
parser.add_argument("test_path", help="the path of test file")
parser.add_argument("-c", "--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g", "--gpu", help="the id of gpu, the default is 0", default=0, type=int)
args = parser.parse_args()

model_path = args.model_path
test_path = args.test_path
output_path = None
gpu_config = "/gpu:"+str(args.gpu)
emb_path = args.char_emb
num_steps = 200  # it must consist with the train

start_time = time.time()

print "preparing test data"
X_test, X_left_test, X_right_test, X_pos_test, X_lpos_test, X_rpos_test, X_rel_test, X_dis_test = helper.getTest(test_path=test_path, seq_max_len=num_steps)

test_data = {}
test_data['char'] = X_test
test_data['left'] = X_left_test
test_data['right'] = X_right_test
test_data['pos'] = X_pos_test
test_data['lpos'] = X_lpos_test
test_data['rpos'] = X_rpos_test
test_data['rel'] = X_rel_test
test_data['dis'] = X_dis_test

char2id, id2char = helper.loadMap("char2id")
pos2id, id2pos = helper.loadMap("pos2id")
label2id, id2label = helper.loadMap("label2id")

num_chars = len(id2char.keys())
num_poses = len(id2pos.keys())
num_classes = len(id2label.keys())
num_dises = 250

if emb_path != None:
    embedding_matrix = helper.getEmbedding(emb_path)
else:
    embedding_matrix = None

name_list = [2, 3, 4, 5, 6, 7, 8]

# voting function, for the i-th word, we select the tag that most models predicted.
def voting(pred): 
    length = len(pred[0])
    new_pred = []
    a = None
    for i in range(length):
        # i-th sample in testing set
        for j in range(len(pred)):
            # j-th model
            if j == 0:
                a = []
                for s in range(len(pred[j][i])):
                    a.append([])
            for s, k in enumerate(pred[j][i]):
                a[s].append(k)
            if j == len(pred)-1:
                # get the label most models predicted
                na = []
                for s in a:
                    # print (s)
                    count = np.bincount(np.array(s))
                    max_value = np.argmax(count)
                    na.append(max_value)
                new_pred.append(copy.deepcopy(na))
    return new_pred


config = tf.ConfigProto(allow_soft_placement=True)
preds = [] # i-th element is the result of i-th model
for name in name_list:
    with tf.Session(config=config) as sess:
        with tf.device(gpu_config):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model_{}".format(name), reuse=None, initializer=initializer):
                model = BILSTM_CRF(num_chars=num_chars, num_poses=num_poses, num_dises=num_dises, num_classes=num_classes, num_steps=num_steps,
                                       embedding_matrix=embedding_matrix, is_training=False)
            print "loading model parameter"
            saver = tf.train.Saver()
            save_path = 'param_run/run_{}/model.ckpt-278'.format(name)
            saver.restore(sess, save_path) # load parameters from the "save_path".

            print "testing"
            predlines = model.test(sess, test_data, 'out_file')

            preds.append(predlines)

            end_time = time.time()
            print "time used %f(hour)" % ((end_time - start_time) / 3600)

pred = voting(preds)
# save results in test.label
with open('test.label','w') as f:
    for p in pred:
        y_pred = [id2label[val] for val in p]
        f.write(' '.join(y_pred)+'\n')

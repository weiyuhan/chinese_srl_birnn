# encoding:utf-8
import re
import os
import csv
import time
import pickle
import numpy as np
import pandas as pd

csv_name = ["char", "left", "right", "pos", "lpos", "rpos", "rel", "dis", "label"]

def getEmbedding(infile_path="embedding"):
    char2id, id_char = loadMap("char2id")
    row_index = 0
    with open(infile_path, "rb") as infile:
        for row in infile:
            row = row.strip()
            row_index += 1
            if row_index == 1:
                num_chars = int(row.split()[0])
                emb_dim = int(row.split()[1])
                emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
                continue
            items = row.split()
            char = items[0]
            emb_vec = [float(val) for val in items[1:]]
            if char in char2id:
                emb_matrix[char2id[char]] = emb_vec
    return emb_matrix


def nextBatch(dataSet, start_index, batch_size=128):
    X = dataSet['char']
    X_left = dataSet['left']
    X_right = dataSet['right']
    X_pos = dataSet['pos']
    X_lpos = dataSet['lpos']
    X_rpos = dataSet['rpos']
    X_rel = dataSet['rel']
    X_dis = dataSet['dis']
    y = dataSet['label']

    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    X_left_batch = list(X_left[start_index:min(last_index, len(X))])
    X_right_batch = list(X_right[start_index:min(last_index, len(X))])

    X_pos_batch = list(X_pos[start_index:min(last_index, len(X))])
    X_lpos_batch = list(X_lpos[start_index:min(last_index, len(X))])
    X_rpos_batch = list(X_rpos[start_index:min(last_index, len(X))])

    X_rel_batch = list(X_rel[start_index:min(last_index, len(X))])
    X_dis_batch = list(X_dis[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            X_left_batch.append(X_left[index])
            X_right_batch.append(X_right[index])
            X_pos_batch.append(X_pos[index])
            X_lpos_batch.append(X_lpos[index])
            X_rpos_batch.append(X_rpos[index])
            X_rel_batch.append(X_rel[index])
            X_dis_batch.append(X_dis[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    X_left_batch = np.array(X_left_batch)
    X_right_batch = np.array(X_right_batch)
    X_pos_batch = np.array(X_pos_batch)
    X_lpos_batch = np.array(X_lpos_batch)
    X_rpos_batch = np.array(X_rpos_batch)
    X_rel_batch = np.array(X_rel_batch)
    X_dis_batch = np.array(X_dis_batch)
    y_batch = np.array(y_batch)

    batches = {}
    batches['char'] = X
    batches['left'] = X_left
    batches['right'] = X_right
    batches['pos'] = X_pos
    batches['lpos'] = X_lpos
    batches['rpos'] = X_rpos
    batches['rel'] = X_rel
    batches['dis'] = X_dis
    batches['label'] = y
    return batches

def nextRandomBatch(dataSet, batch_size=128):
    X = dataSet['char']
    X_left = dataSet['left']
    X_right = dataSet['right']
    X_pos = dataSet['pos']
    X_lpos = dataSet['lpos']
    X_rpos = dataSet['rpos']
    X_rel = dataSet['rel']
    X_dis = dataSet['dis']
    y = dataSet['label']

    X_batch = []
    X_left_batch = []
    X_right_batch = []
    X_pos_batch = []
    X_lpos_batch = []
    X_rpos_batch = []
    X_rel_batch = []
    X_dis_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        X_left_batch.append(X_left[index])
        X_right_batch.append(X_right[index])
        X_pos_batch.append(X_pos[index])
        X_lpos_batch.append(X_lpos[index])
        X_rpos_batch.append(X_rpos[index])
        X_rel_batch.append(X_rel[index])
        X_dis_batch.append(X_dis[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    X_left_batch = np.array(X_left_batch)
    X_right_batch = np.array(X_right_batch)
    X_pos_batch = np.array(X_pos_batch)
    X_lpos_batch = np.array(X_lpos_batch)
    X_rpos_batch = np.array(X_rpos_batch)
    X_rel_batch = np.array(X_rel_batch)
    X_dis_batch = np.array(X_dis_batch)
    y_batch = np.array(y_batch)

    batches = {}
    batches['char'] = X
    batches['left'] = X_left
    batches['right'] = X_right
    batches['pos'] = X_pos
    batches['lpos'] = X_lpos
    batches['rpos'] = X_rpos
    batches['rel'] = X_rel
    batches['dis'] = X_dis
    batches['label'] = y
    return batches

# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def prepare(chars, lefts, rights, poss, lposs, rposs, rels, diss, labels, seq_max_len, is_padding=True):
    X = []
    X_left = []
    X_right = []
    X_pos = []
    X_lpos = []
    X_rpos = []
    X_rel = []
    X_dis = []
    y = []

    tmp_x = []
    tmp_left = []
    tmp_right = []
    tmp_pos = []
    tmp_lpos = []
    tmp_rpos =[]
    tmp_rel = []
    tmp_dis = []
    tmp_y = []

    for record in zip(chars, lefts, rights, poss, lposs, rposs, rels, diss, labels):
        c = record[0]
        lc = record[1]
        rc = record[2]
        p = record[3]
        lp = record[4]
        rp = record[5]
        rl = record[6]
        d = record[7]
        l = record[2]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                X_left.append(tmp_left)
                X_right.append(tmp_right)
                X_pos.append(tmp_pos)
                X_lpos.append(tmp_lpos)
                X_rpos.append(tmp_rpos)
                X_rel.append(tmp_rel)
                X_dis.append(tmp_dis)
                y.append(tmp_y)
            tmp_x = []
            tmp_left = []
            tmp_right = []
            tmp_pos = []
            tmp_lpos = []
            tmp_rpos =[]
            tmp_rel = []
            tmp_dis = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_left.append(lc)
            tmp_right.append(rc)
            tmp_pos.append(p)
            tmp_lpos.append(lp)
            tmp_rpos.append(rp)
            tmp_rel.append(rl)
            tmp_dis.append(d)
            tmp_y.append(l)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
        X_left = np.array(padding(X_left, seq_max_len))
        X_right = np.array(padding(X_right, seq_max_len))
        X_pos = np.array(padding(X_pos, seq_max_len))
        X_lpos = np.array(padding(X_lpos, seq_max_len))
        X_rpos = np.array(padding(X_rpos, seq_max_len))
        X_rel = np.array(padding(X_rel, seq_max_len))
        X_dis = np.array(padding(X_dis, seq_max_len))
    else:
        X = np.array(X)
        X_left = np.array(X_left)
        X_right = np.array(X_right)
        X_pos = np.array(X_pos)
        X_lpos = np.array(X_lpos)
        X_rpos = np.array(X_rpos)
        X_rel = np.array(X_rel)
        X_dis = np.array(X_dis)
    y = np.array(padding(y, seq_max_len))

    return X, X_left, X_right, X_pos, X_lpos, X_rpos, X_rel, X_dis, y


def loadMap(token2id_filepath):
    if not os.path.isfile(token2id_filepath):
        print "file not exist, building map"
        buildMap()

    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip().decode("utf-8")
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def saveMap(id2char, id2pos, id2label):
    with open("char2id", "wb") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx) + "\r\n")
    with open("pos2id", "wb") as outfile:
        for idx in id2pos:
            outfile.write(id2pos[idx] + "\t" + str(idx) + "\r\n")
    with open("label2id", "wb") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\r\n")
    print "saved map between token and id"


def buildMap(train_path="train.in"):
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=csv_name)
    
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    poses = list(set(df_train["pos"][df_train["pos"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))

    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    pos2id = dict(zip(poses, range(1, len(poses) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))

    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2pos = dict(zip(range(1, len(poses) + 1), poses))
    id2label = dict(zip(range(1, len(labels) + 1), labels))

    id2char[0] = "<PAD>"
    id2pos[0] = "<PAD>"
    id2label[0] = "<PAD>"

    char2id["<PAD>"] = 0
    pos2id["<PAD>"] = 0
    label2id["<PAD>"] = 0

    id2char[len(chars) + 1] = "<NEW>"
    id2pos[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1
    pos2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, id2pos, id2label)

    return char2id, id2char, pos2id, id2pos, label2id, id2label


def getTrain(train_path, val_path, train_val_ratio=0.99, use_custom_val=False, seq_max_len=200):
    char2id, id2char, pos2id, id2pos, label2id, id2label = buildMap(train_path)
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=csv_name)

    # map the char and label into id
    df_train["char_id"] = df_train.char.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["left_id"] = df_train.left.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["right_id"] = df_train.right.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["rel_id"] = df_train.rel.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    
    df_train["pos_id"] = df_train.pos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])
    df_train["lpos_id"] = df_train.lpos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])
    df_train["rpos_id"] = df_train.rpos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])
    
    df_train["dis_id"] = df_train.rpos.map(lambda x: int(x))
    
    df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    # convert the data in maxtrix
    X, X_pos, y = prepare(df_train["char_id"], df_train["left_id"], df_train["right_id"],
        df_train["pos_id"], df_train["lpos_id"], df_train["rpos_id"],
        df_train["rel_id"], df_train["dis_id"], df_train["label_id"], seq_max_len)

    # shuffle the samples
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    X_left = X_left[indexs]
    X_right = X_right[indexs]
    X_pos = X_pos[indexs]
    X_lpos = X_lpos[indexs]
    X_rpos = X_rpos[indexs]
    X_rel = X_rel[indexs]
    X_dis = X_dis[indexs]
    y = y[indexs]

    if val_path != None:
        X_train = X
        X_left_train = X_left
        X_right_train = X_right
        X_pos_train = X_pos
        X_lpos_train = X_lpos
        X_rpos_train = X_rpos
        X_rel_train = X_rel
        X_dis_train = X_dis
        y_train = y
        X_val, X_left_val, X_right_val, X_pos_val, X_lpos_val, X_rpos_val, X_rel_val, X_dis_val, y_val = getTest(val_path, is_validation=True, seq_max_len=seq_max_len)
    else:
        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        X_left_train = X_left[:int(num_samples * train_val_ratio)]
        X_right_train = X_right[:int(num_samples * train_val_ratio)]
        X_pos_train = X_pos[:int(num_samples * train_val_ratio)]
        X_lpos_train = X_lpos[:int(num_samples * train_val_ratio)]
        X_rpos_train = X_rpos[:int(num_samples * train_val_ratio)]
        X_rel_train = X_rel[:int(num_samples * train_val_ratio)]
        X_dis_train = X_dis[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        
        X_val = X[int(num_samples * train_val_ratio):]
        X_left_val = X_left[int(num_samples * train_val_ratio):]
        X_right_val = X_right[int(num_samples * train_val_ratio):]
        X_pos_val = X_pos[int(num_samples * train_val_ratio):]
        X_lpos_val = X_lpos[int(num_samples * train_val_ratio):]
        X_rpos_val = X_rpos[int(num_samples * train_val_ratio):]
        X_rel_val = X_rel[int(num_samples * train_val_ratio):]
        X_dis_val = X_dis[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

    print "train size: %d, validation size: %d" % (len(X_train), len(y_val))

    train_data['char'] = X_train
    train_data['left'] = X_left_train
    train_data['right'] = X_right_train
    train_data['pos'] = X_pos_train
    train_data['lpos'] = X_lpos_train
    train_data['rpos'] = X_rpos_train
    train_data['rel'] = X_rel_train
    train_data['dis'] = X_dis_train
    train_data['label'] = y_train

    val_data['char'] = X_val
    val_data['left'] = X_left_val
    val_data['right'] = X_right_val
    val_data['pos'] = X_pos_val
    val_data['lpos'] = X_lpos_val
    val_data['rpos'] = X_rpos_val
    val_data['rel'] = X_rel_val
    val_data['dis'] = X_dis_val
    val_data['label'] = y_val

    return train_data, val_data

def getTest(test_path="test.in", is_validation=False, seq_max_len=200):
    char2id, id2char = loadMap("char2id")
    pos2id, id2pos = loadMap("pos2id")
    label2id, id2label = loadMap("label2id")

    df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=csv_name)

    def mapFunc(x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x.decode("utf-8") not in token2id:
            return token2id["<NEW>"]
        else:
            return token2id[x.decode("utf-8")]

    df_test["char_id"] = df_test.char.map(lambda x: mapFunc(x, char2id))
    df_test["left_id"] = df_test.left.map(lambda x: mapFunc(x, char2id))
    df_test["right_id"] = df_test.right.map(lambda x: mapFunc(x, char2id))
    df_test["rel_id"] = df_test.rel.map(lambda x: mapFunc(x, char2id))
    
    df_test["pos_id"] = df_test.pos.map(lambda x: mapFunc(x, pos2id))
    df_test["lpos_id"] = df_test.lpos.map(lambda x: mapFunc(x, pos2id))
    df_test["rpos_id"] = df_test.rpos.map(lambda x: mapFunc(x, pos2id))
    
    df_test["dis_id"] = df_test.dis.map(lambda x: int(x))
    
    df_test["label_id"] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    X_test, X_left_test, X_right_test, X_pos_test, X_lpos_test, X_rpos_test, X_rel_test, X_dis_test, y_test = prepare(
        df_test["char_id"], df_test["left_id"], df_test["right_id"], 
        df_test["pos_id"], df_test["lpos_id"], df_test["rpos_id"], 
        df_test["rel_id"], df_test["dis_id"], df_test["label_id"], seq_max_len)
    if is_validation:
        return X_test, X_left_test, X_right_test, X_pos_test, X_lpos_test, X_rpos_test, X_rel_test, X_dis_test, y_test
    else:
        return X_test, X_left_test, X_right_test, X_pos_test, X_lpos_test, X_rpos_test, X_rel_test, X_dis_test


def getTransition(y_train_batch):
    transition_batch = []
    for m in range(len(y_train_batch)):
        y = [5] + list(y_train_batch[m]) + [0]
        for t in range(len(y)):
            if t + 1 == len(y):
                continue
            i = y[t]
            j = y[t + 1]
            if i == 0:
                break
            transition_batch.append(i * 6 + j)
    transition_batch = np.array(transition_batch)
    return transition_batch

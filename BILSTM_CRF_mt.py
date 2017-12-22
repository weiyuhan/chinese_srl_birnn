import math
import helper
import numpy as np
import tensorflow as tf

class BILSTM_CRF(object):
    
    def __init__(self, num_chars, num_poses, num_dises, num_classes, num_steps=200, num_epochs=100, embedding_matrix=None, is_training=True, is_crf=False, weight=False):
        # Parameter
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 20
        self.num_layers = 1   
        self.emb_dim = 50 #char, left, right, rel
        self.pos_dim = 25 #pos, lpos, rpos
        self.dis_dim = 25 #dis
        self.hidden_dim = 300
        self.filter_sizes = [3]
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_poses = num_poses
        self.num_dises = num_dises
        self.num_classes = num_classes
        self.is_crf = is_crf
        
        # placeholder for feature
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.lefts = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rights = tf.placeholder(tf.int32, [None, self.num_steps])
        self.poses = tf.placeholder(tf.int32, [None, self.num_steps])
        self.lposes = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rposes = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rels = tf.placeholder(tf.int32, [None, self.num_steps])
        self.dises = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        
        # word embedding
        if embedding_matrix != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])

        #input word feature vector
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.lefts_emb = tf.nn.embedding_lookup(self.embedding, self.lefts)
        self.rights_emb = tf.nn.embedding_lookup(self.embedding, self.rights)
        self.rels_emb = tf.nn.embedding_lookup(self.embedding, self.rels)

        #pos vecor
        self.pos_embedding = tf.get_variable("pos_embedding", [self.num_poses, self.pos_dim])
        self.pos_emb = tf.nn.embedding_lookup(self.pos_embedding, self.poses)
        self.lpos_emb = tf.nn.embedding_lookup(self.pos_embedding, self.lposes)
        self.rpos_emb = tf.nn.embedding_lookup(self.pos_embedding, self.rposes)

        #dis vector
        self.dis_embedding = tf.get_variable("dis_embedding", [self.num_dises, self.dis_dim])
        self.dis_emb = tf.nn.embedding_lookup(self.dis_embedding, self.dises)

        #nonlinear layer
        self.inputs_emb = tf.concat([self.inputs_emb, 
            self.pos_emb, self.rels_emb, self.dis_emb], axis=2)
        self.inputs_emb = tf.tanh(self.inputs_emb)

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), axis=1)
        self.length = tf.cast(self.length, tf.int32)  
        
        #birnn
        self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            self.inputs_emb, 
            dtype=tf.float32,
            sequence_length=self.length
        )
        
        # softmax
        self.outputs = tf.reshape(tf.concat(axis=2, values=self.outputs), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.logits = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])

        if not is_crf: #not_crf
            self.tags_scores = self.logits
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)
            losses = tf.boolean_mask(self.loss, mask)
            self.loss = tf.reduce_mean(losses)
            self.trans_params = self.length
        else:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.targets, self.length)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        
        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) 

    #train process
    def train(self, sess, save_file, train_data, val_data):
        saver = tf.train.Saver(max_to_keep=3)

        X_train = train_data['char']
        X_left_train = train_data['left']
        X_right_train = train_data['right']
        X_pos_train = train_data['pos']
        X_lpos_train = train_data['lpos']
        X_rpos_train = train_data['rpos']
        X_rel_train = train_data['rel']
        X_dis_train = train_data['dis']
        y_train = train_data['label']

        X_val = val_data['char']
        X_left_val = val_data['left']
        X_right_val = val_data['right']
        X_pos_val = val_data['pos']
        X_lpos_val = val_data['lpos']
        X_rpos_val = val_data['rpos']
        X_rel_val = val_data['rel']
        X_dis_val = val_data['dis']
        y_val = val_data['label']

        char2id, id2char = helper.loadMap("char2id")
        pos2id, id2pos = helper.loadMap("pos2id")
        label2id, id2label = helper.loadMap("label2id")

        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)  
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)     
        
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            X_left_train = X_left_train[sh_index]
            X_right_train = X_right_train[sh_index]
            X_pos_train = X_pos_train[sh_index]
            X_lpos_train = X_lpos_train[sh_index]
            X_rpos_train = X_rpos_train[sh_index]
            X_rel_train = X_rel_train[sh_index]
            X_dis_train = X_dis_train[sh_index]
            y_train = y_train[sh_index]

            train_data['char'] = X_train
            train_data['left'] = X_left_train
            train_data['right'] = X_right_train
            train_data['pos'] = X_pos_train
            train_data['lpos'] = X_lpos_train
            train_data['rpos'] = X_rpos_train
            train_data['rel'] = X_rel_train
            train_data['dis'] = X_dis_train
            train_data['label'] = y_train

            print "current epoch: %d" % (epoch)
            for iteration in range(num_iterations):
                # train 
                #get batch
                train_batches = helper.nextBatch(train_data, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                X_train_batch = train_batches['char']
                X_left_train_batch = train_batches['left']
                X_right_train_batch = train_batches['right']
                X_pos_train_batch = train_batches['pos']
                X_lpos_train_batch = train_batches['lpos']
                X_rpos_train_batch = train_batches['rpos']
                X_rel_train_batch = train_batches['rel']
                X_dis_train_batch = train_batches['dis']
                y_train_batch = train_batches['label']
                
                _, loss_train, length, train_summary, logits, trans_params =\
                    sess.run([
                        self.optimizer, 
                        self.loss, 
                        self.length,
                        self.train_summary,
                        self.logits,
                        self.trans_params,
                    ], 
                    feed_dict={
                        self.inputs:X_train_batch,
                        self.lefts:X_left_train_batch,
                        self.rights:X_right_train_batch,
                        self.poses:X_pos_train_batch,
                        self.lposes:X_lpos_train_batch,
                        self.rposes:X_rpos_train_batch,
                        self.rels:X_rel_train_batch,
                        self.dises:X_dis_train_batch,
                        self.targets:y_train_batch 
                        # self.targets_weight:y_train_weight_batch
                    })
                # print (len(length))

                # calc f1 for the whole dev set
                if epoch > 0 and iteration == num_iterations -1:
                    num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))
                    preds_lines = []
                    for val_iteration in range(num_val_iterations):
                        val_batches = helper.nextBatch(val_data, start_index=val_iteration * self.batch_size, batch_size=self.batch_size)
                        X_val_batch = val_batches['char']
                        X_left_val_batch = val_batches['left']
                        X_right_val_batch = val_batches['right']
                        X_pos_val_batch = val_batches['pos']
                        X_lpos_val_batch = val_batches['lpos']
                        X_rpos_val_batch = val_batches['rpos']
                        X_rel_val_batch = val_batches['rel']
                        X_dis_val_batch = val_batches['dis']
                        y_val_batch = val_batches['label']

                        loss_val, length, val_summary, logits, trans_params =\
                            sess.run([
                                self.loss, 
                                self.length,
                                self.val_summary,
                                self.logits,
                                self.trans_params,
                            ], 
                            feed_dict={
                                self.inputs:X_val_batch,
                                self.lefts:X_left_val_batch,
                                self.rights:X_right_val_batch,
                                self.poses:X_pos_val_batch,
                                self.lposes:X_lpos_val_batch,
                                self.rposes:X_rpos_val_batch,
                                self.rels:X_rel_val_batch,
                                self.dises:X_dis_val_batch,
                                self.targets:y_val_batch 
                                # self.targets_weight:y_val_weight_batch
                            })
                    
                        predicts_val = self.viterbi(logits, trans_params, length, predict_size=self.batch_size)
                        preds_lines.extend(predicts_val)
                    preds_lines = preds_lines[:len(y_val)]
                    recall_val, precision_val, f1_val, errors = helper.calc_f1(preds_lines, id2label, 'cpbdev.txt', 'validation.out')
                    if f1_val > self.max_f1:
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file + '/model.ckpt', global_step=iteration)
                        helper.calc_f1(preds_lines, id2label, 'cpbdev.txt', 'validation.out.best')
                        print "saved the best model with f1: %.5f" % (self.max_f1)
                    print "valid precision: %.5f, valid recall: %.5f, valid f1: %.5f, errors: %5d" % (precision_val, recall_val, f1_val, errors)

    #test process
    def test(self, sess, test_data, output_path):
        #data
        X_test = test_data['char']
        X_left_test = test_data['left']
        X_right_test = test_data['right']
        X_pos_test = test_data['pos']
        X_lpos_test = test_data['lpos']
        X_rpos_test = test_data['rpos']
        X_rel_test = test_data['rel']
        X_dis_test = test_data['dis']
        #dictionary
        char2id, id2char = helper.loadMap("char2id")
        pos2id, id2pos = helper.loadMap("pos2id")
        label2id, id2label = helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print "number of iteration: " + str(num_iterations)
        with open(output_path, "wb") as outfile:
            for i in range(num_iterations):
                print "iteration: " + str(i + 1)
                results = []
                #get batch
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_left_test_batch = X_left_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_right_test_batch = X_right_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_pos_test_batch = X_pos_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_lpos_test_batch = X_lpos_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_rpos_test_batch = X_rpos_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_rel_test_batch = X_rel_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_dis_test_batch = X_dis_test[i * self.batch_size : (i + 1) * self.batch_size]
                # left seqtence less than batch size, use [0] as seq
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_left_test_batch = list(X_left_test_batch)
                    X_right_test_batch = list(X_right_test_batch)
                    X_pos_test_batch = list(X_pos_test_batch)
                    X_lpos_test_batch = list(X_lpos_test_batch)
                    X_rpos_test_batch = list(X_rpos_test_batch)
                    X_rel_test_batch = list(X_rel_test_batch)
                    X_dis_test_batch = list(X_dis_test_batch)
                    
                    last_size = len(X_test_batch)

                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_left_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_right_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_pos_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_lpos_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_rpos_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_rel_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_dis_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    
                    X_test_batch = np.array(X_test_batch)
                    X_left_test_batch = np.array(X_left_test_batch)
                    X_right_test_batch = np.array(X_right_test_batch)
                    X_pos_test_batch = np.array(X_pos_test_batch) 
                    X_lpos_test_batch = np.array(X_lpos_test_batch)
                    X_rpos_test_batch = np.array(X_rpos_test_batch)
                    X_rel_test_batch = np.array(X_rel_test_batch)
                    X_dis_test_batch = np.array(X_dis_test_batch)

                    test_batches = {}
                    test_batches['char'] = X_test_batch
                    test_batches['left'] = X_left_test_batch
                    test_batches['right'] = X_right_test_batch
                    test_batches['pos'] = X_pos_test_batch
                    test_batches['lpos'] = X_lpos_test_batch
                    test_batches['rpos'] = X_rpos_test_batch
                    test_batches['rel'] = X_rel_test_batch
                    test_batches['dis'] = X_dis_test_batch
                    results = self.predictBatch(sess, test_batches, id2label)
                    results = results[:last_size]
                else:  # next batch
                    X_test_batch = np.array(X_test_batch)
                    X_left_test_batch = np.array(X_left_test_batch)
                    X_right_test_batch = np.array(X_right_test_batch)
                    X_pos_test_batch = np.array(X_pos_test_batch) 
                    X_lpos_test_batch = np.array(X_lpos_test_batch)
                    X_rpos_test_batch = np.array(X_rpos_test_batch)
                    X_rel_test_batch = np.array(X_rel_test_batch)
                    X_dis_test_batch = np.array(X_dis_test_batch)

                    test_batches = {}
                    test_batches['char'] = X_test_batch
                    test_batches['left'] = X_left_test_batch
                    test_batches['right'] = X_right_test_batch
                    test_batches['pos'] = X_pos_test_batch
                    test_batches['lpos'] = X_lpos_test_batch
                    test_batches['rpos'] = X_rpos_test_batch
                    test_batches['rel'] = X_rel_test_batch
                    test_batches['dis'] = X_dis_test_batch
                    
                    results = self.predictBatch(sess, test_batches, id2label)
    #get result using viterbi when crf, argmax when not_crf
    def viterbi(self, logits, trans_params, lengths, predict_size=128):
        viterbi_sequences = []

        # iterate over the sentences because no batching in vitervi_decode
        if not self.is_crf:
            for logit, sequence_length in zip(logits, lengths):
                if sequence_length == 0:
                    viterbi_sequences += [[]]
                    continue
                logit = logit[:sequence_length]
                logit = tf.nn.softmax(logit)
                labels_pred = tf.cast(tf.argmax(logit, axis=-1), tf.int32)
                viterbi_sequences += [labels_pred.eval()]
            return viterbi_sequences

        for logit, sequence_length in zip(logits, lengths):
            if sequence_length == 0:
                viterbi_sequences += [[]]
                continue
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    #predict a result for a batch
    def predictBatch(self, sess, batches, id2label):
        results = []
        
        X = batches['char']
        X_left = batches['left']
        X_right = batches['right']
        X_pos = batches['pos']
        X_lpos = batches['lpos']
        X_rpos = batches['rpos']
        X_rel = batches['rel']
        X_dis = batches['dis']
        
        length, logits, trans_params = sess.run([self.length, self.logits, self.trans_params], 
            feed_dict={
                self.inputs:X,
                self.lefts:X_left,
                self.rights:X_right,
                self.poses:X_pos,
                self.lposes:X_lpos,
                self.rposes:X_rpos,
                self.rels:X_rel,
                self.dises:X_dis
            })
        predicts = self.viterbi(logits, trans_params, length, self.batch_size)
        for i in range(len(predicts)):
            y_pred = [id2label[val] for val in predicts[i]]
            results.append(y_pred)
        return results
    #show a f1 at trainning
    def evaluate(self, y_true, y_pred,id2char, id2label):
        hit_num = 0
        pred_num = 0
        true_num = 0
        for i in range(len(y_true)):
            # print (y_true[i])
            # print (y_pred[i])
            y = [str(id2label[val].encode("utf-8")) for val in y_true[i]]
            y_hat = [str(id2label[val].encode("utf-8")) for val in y_pred[i]]
            for t in range(len(y_hat)):
                if y[t] == y_hat[t] and y_hat[t] != 'O':
                    hit_num += 1 
                if y_hat[t] != '<PAD>' and y_hat[t] != 'O':
                    pred_num += 1
                if y[t] != '<PAD>' and y[t] != 'O':
                    true_num +=1 
        return hit_num, pred_num, true_num 
    #show a f1 at trainning
    def caculate(self, hit_num, pred_num, true_num):
        precision = -1.0;
        recall = -1.0
        f1 = -1.0
        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1

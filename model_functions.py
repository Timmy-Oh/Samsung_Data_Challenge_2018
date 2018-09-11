import tensorflow as tf
import numpy as np

import config 

def build_model(config):
    tf.reset_default_graph()
    #inputs
    inp_cate = tf.placeholder(dtype=tf.int32, shape=[None, config.cate_len], name='categorical_input_layer')
    inp_cont = tf.placeholder(dtype=tf.float32, shape=[None, config.cont_len], name='continuous_input_layer')
    inp_cont_scale = tf.placeholder(dtype=tf.float32, shape=[None, config.cont_len], name='continuous_scaled_input_layer')
    inp_cont_tf = tf.placeholder(dtype=tf.int32, shape=[None, config.cont_len], name='continuous_tf_input_layer')
    inp_cate_y = tf.placeholder(dtype=tf.int32, shape=[None, config.cate_len], name='categorical_input_y_layer')
    inp_cont_y = tf.placeholder(dtype=tf.float32, shape=[None, config.cont_len], name='continuous_input_y_layer')
    kp = tf.placeholder(dtype=tf.float32, shape=[], name='kp')
    
    #embedding
    cate_oh = [tf.one_hot(inp_cate[:, i], depth=config.cate_lens[i]) for i in range(len(config.cate_cols))]
    cate_oh_y = [tf.one_hot(inp_cate_y[:, i], depth=config.cate_lens[i]) for i in range(len(config.cate_cols))]
    concat_cate_y = tf.concat(cate_oh_y, axis=-1)
    
    oh_cont_tf = [tf.one_hot(inp_cont_tf[:, i], depth=2) for i in range(config.cont_len)]
    
    concat_cate = tf.concat(cate_oh, axis=-1)
    concat_cont_tf = tf.concat(oh_cont_tf, axis=-1)
    concat_all = tf.concat([concat_cate, inp_cont, concat_cont_tf], axis=-1)
    
    #hidden layer
    l1_nd = tf.contrib.layers.fully_connected(concat_all, config.l1_size)
    l1 = tf.nn.dropout(l1_nd, kp)
    l12 = tf.contrib.layers.fully_connected(l1, config.l2_size)
    l21 = tf.contrib.layers.fully_connected(concat_all, config.l1_size)
    l22 = tf.contrib.layers.fully_connected(l21, config.l2_size)
    

    #logits
    logit_cates = [tf.contrib.layers.fully_connected(l1, length, activation_fn = None) for length in config.cate_lens]
    pred_cates = [tf.nn.softmax(logit) for logit in logit_cates]
    pred_cate = tf.concat(pred_cates, axis=-1)
    
    l2 = tf.concat([l12, l22, pred_cate], axis=-1)
    logit_cont1 = tf.contrib.layers.fully_connected(l12, config.cont_out_size, activation_fn = None)
    logit_cont2 = tf.contrib.layers.fully_connected(l22, config.cont_out_size, activation_fn = None)
    logit_cont3 = tf.contrib.layers.fully_connected(l2, config.cont_out_size, activation_fn = None)
    logit_cont = tf.reduce_mean([logit_cont1, logit_cont2, logit_cont3], axis=0)

    pred_cont = logit_cont

    #losses
    loss_cate_ops = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inp_cate_y[:,idx], logits=logit_cates[idx])) for idx in range(11)]
    loss_cate_op = tf.reduce_sum(loss_cate_ops, axis=-1)
    loss_cont_op = tf.losses.mean_squared_error(inp_cont_y, pred_cont)
    
    loss_op = loss_cate_op + loss_cont_op*2
    
    #opt
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    return loss_op, loss_cate_op, loss_cont_op, train_op, inp_cate, inp_cont, inp_cont_scale, inp_cont_tf, inp_cate_y, inp_cont_y, pred_cate, pred_cont, kp

def run_session(sess, max_step, datas, config, mdl, mode=1):
    total_loss = np.array([0.0, 0.0, 0.0])
    loss_op, loss_cate_op, loss_cont_op, train_op, inp_cate, inp_cont, inp_cont_scale, inp_cont_tf, inp_cate_y, inp_cont_y, pred_cate, pred_cont, kp = mdl

    #train
    if mode ==1:
        drop_mask = np.random.rand(datas[0].values.shape[0], datas[0].values.shape[1])> (1- config.keep_prop)
        droped_cate = datas[0].values.copy()
        droped_cate = droped_cate * drop_mask
        
        cont_mask = np.random.rand(datas[1].values.shape[0], datas[1].values.shape[1])< (1- config.keep_prop)
        droped_cont = datas[1].values.copy()
        droped_cont[cont_mask] = config.replace_cont
        droped_cont_scaled = (droped_cont - config.cont_min)/(config.cont_max - config.cont_min)
        
        for step in range(max_step):
            start_idx = step * config.batch_size
            loss_ = sess.run([loss_op, loss_cate_op, loss_cont_op, train_op], feed_dict = {inp_cate:droped_cate[start_idx:start_idx+config.batch_size], 
                                                                  inp_cont:droped_cont[start_idx:start_idx+config.batch_size],
                                                                                           inp_cont_scale:droped_cont_scaled[start_idx:start_idx+config.batch_size],
                                                                                           inp_cont_tf:cont_mask[start_idx:start_idx+config.batch_size],
                                                                  inp_cate_y:datas[0].values[start_idx:start_idx+config.batch_size],
                                                                  inp_cont_y:datas[1].values[start_idx:start_idx+config.batch_size],
                                                                  kp:0.7})
            
            total_loss += loss_[:-1]
        total_loss /= max_step
        return total_loss 
    
    #validation
    elif mode == 2:
        droped_valid_cate = datas[0].values * config.vaild_drop_mask
        droped_valid_cont = datas[1].values.copy()
        droped_valid_cont[config.vaild_cont_mask] = config.replace_cont
        droped_valid_cont_scaled = (droped_valid_cont - config.cont_min)/(config.cont_max - config.cont_min)
        
        for step in range(max_step):
            start_idx = step * config.batch_size
            loss_ = sess.run([loss_op, loss_cate_op, loss_cont_op], feed_dict = {inp_cate:droped_valid_cate[start_idx:start_idx+config.batch_size], 
                                                   inp_cont:droped_valid_cont[start_idx:start_idx+config.batch_size], 
                                                                                 inp_cont_scale:droped_valid_cont_scaled[start_idx:start_idx+config.batch_size],
                                                                                 inp_cont_tf:config.vaild_cont_mask[start_idx:start_idx+config.batch_size],
                                                                                 
                                                   inp_cate_y:datas[0].values[start_idx:start_idx+config.batch_size],
                                                   inp_cont_y:datas[1].values[start_idx:start_idx+config.batch_size],
                                                   kp:1.0})
            
            total_loss += loss_
        total_loss /= max_step
        return total_loss 
    
    #valid_test
    elif mode == 3:
        droped_valid_cate = datas[0].values * config.vaild_test_drop_mask
        droped_valid_cont = datas[1].values.copy()
        droped_valid_cont[config.vaild_test_cont_mask] = config.replace_cont
        droped_valid_cont_scaled = (droped_valid_cont - config.cont_min)/(config.cont_max - config.cont_min)
        
        
        preds_cate, preds_cont = [], []
        for step in range(max_step):
            start_idx = step * config.batch_size
            pred_cate_, pred_cont_ = sess.run([pred_cate, pred_cont], 
                                              feed_dict = {inp_cate:droped_valid_cate[start_idx:start_idx+config.batch_size], 
                                                           inp_cont:droped_valid_cont[start_idx:start_idx+config.batch_size],
                                                           inp_cont_scale:droped_valid_cont_scaled[start_idx:start_idx+config.batch_size],
                                                           inp_cont_tf:config.vaild_test_cont_mask[start_idx:start_idx+config.batch_size],
                                                           kp:1.0})
            preds_cate.append(pred_cate_)
            preds_cont.append(pred_cont_)
            
        return preds_cate, preds_cont
    
    #test
    elif mode == 4:
        test_cate = datas[0].values
        test_cont = datas[1].values.copy()
        test_cont_scaled = (test_cont - config.cont_min)/(config.cont_max - config.cont_min)
        
        
        preds_cate, preds_cont = [], []
        for step in range(max_step):
            start_idx = step * config.batch_size
            pred_cate_, pred_cont_ = sess.run([pred_cate, pred_cont], 
                                              feed_dict = {inp_cate:test_cate[start_idx:start_idx+config.batch_size], 
                                                           inp_cont:test_cont[start_idx:start_idx+config.batch_size],
                                                           inp_cont_scale:test_cont_scaled[start_idx:start_idx+config.batch_size],
                                                           inp_cont_tf:config.test_cont_mask[start_idx:start_idx+config.batch_size],
                                                           kp:1.0})
            preds_cate.append(pred_cate_)
            preds_cont.append(pred_cont_)
            
        return preds_cate, preds_cont
    
    else:
        print('error')
        return None
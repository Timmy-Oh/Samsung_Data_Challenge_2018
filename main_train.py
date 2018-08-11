import pandas as pd
import numpy as np
import pickle as pkl
import tensorflow as tf
import os, math
import config

from model_functions import build_model, run_session


os.makedirs(config.dir_ckpt , exist_ok=True)

raw_df = pd.read_csv('./data/dataset_kor/교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv', engine='python')
raw_df = raw_df[config.test_cols]

for key in config.kv_map.keys():
    raw_df[key] = raw_df[key].apply(lambda x: config.kv_map[key][x])

for mdl_num in range(1,4):
    if mdl_num == 1:
        np.random.seed(seed=1000)
        shuffle_idx = np.random.rand(len(raw_df)) 
        train_df = raw_df[shuffle_idx<= 0.85]
        valid_df = raw_df[shuffle_idx > 0.85]

    elif mdl_num == 2:
        np.random.seed(seed=2000)
        shuffle_idx = np.random.rand(len(raw_df)) 
        train_df = raw_df[shuffle_idx<= 0.85]
        valid_df = raw_df[shuffle_idx > 0.85]

    else:
        train_df = raw_df[:config.train_size]
        valid_df = raw_df[config.train_size:]


    config.path_ckpt = os.path.join(config.dir_ckpt, 'best{}.ckpt'.format(mdl_num))

    train_df_cate = train_df.loc[:,config.cate_cols]
    train_df_cont = train_df.loc[:,config.cont_cols]

    valid_df_cate = valid_df.loc[:,config.cate_cols]
    valid_df_cont = valid_df.loc[:,config.cont_cols]

    np.random.seed(seed=9)
    config.vaild_drop_mask = np.random.rand(valid_df_cate.values.shape[0], valid_df_cate.values.shape[1])>(1- config.keep_prop)
    config.vaild_cont_mask = np.random.rand(valid_df_cont.values.shape[0], valid_df_cont.values.shape[1])<(1- config.keep_prop)

    np.random.seed(seed=19)
    config.vaild_test_drop_mask = np.random.rand(valid_df_cate.values.shape[0], valid_df_cate.values.shape[1])>(1- config.keep_prop)
    config.vaild_test_cont_mask = np.random.rand(valid_df_cont.values.shape[0], valid_df_cont.values.shape[1])<(1- config.keep_prop)

    train_step = math.ceil(len(train_df)/ config.batch_size)
    valid_step = math.ceil(len(valid_df)/ config.batch_size)
    print("Data is Ready...")

    mdl = build_model(config)
    print("Model {} is built...".format(mdl_num))

    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()

    # run session
    with tf.Session() as sess:
        tf.set_random_seed(seed=1991)
        np.random.seed(seed=1991)
        sess.run(init)
    #     saver.restore(sess, config.path_ckpt)
        min_val_loss = 9999
        print("Model {} is training".format(mdl_num))
        for epoch in range(1, config.epochs+1):
            #train
            trn_total_loss_ = run_session(sess, train_step, [train_df_cate, train_df_cont], config, mdl, mode=1)

            #valid
            val_total_loss_ = run_session(sess, valid_step, [valid_df_cate, valid_df_cont], config, mdl, mode=2)

            if config.verbose:
                print("Epoch : {}".format(epoch), end='\t')
                print("Train_loss : {:.6f} / {:.6f} / {:.6f}".format(trn_total_loss_[0], trn_total_loss_[1], trn_total_loss_[2]), end = '\t')
                print("Valid_loss : {:.6f} / {:.6f} / {:.6f}".format(val_total_loss_[0], val_total_loss_[1], val_total_loss_[2]), end = '\t')

            if val_total_loss_[0] < min_val_loss:
                saver.save(sess, config.path_ckpt)
                min_val_loss = val_total_loss_[0]
                if config.verbose:
                    print("Saved")
            else:
                if config.verbose:
                    print("No Saved")
        print("Training Model {} of 3 is Completed...".format(mdl_num))
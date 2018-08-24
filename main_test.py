import pandas as pd
import numpy as np
import pickle as pkl
import tensorflow as tf
import os, math
import config

from model_functions import build_model, run_session



test_df = pd.read_csv('./data/test_kor.csv', engine='python')
test_df = test_df[config.test_cols]
config.test_cate_mask = test_df[config.cate_cols].isna().values
config.test_cont_mask = test_df[config.cont_cols].isna().values

test_df[config.cate_cols] = test_df[config.cate_cols].fillna('N')
test_df[config.cont_cols] = test_df[config.cont_cols].fillna(-1)
for key in config.kv_map.keys():
    test_df[key] = test_df[key].apply(lambda x: config.kv_map[key][x])

mdl = build_model(config)
print("Model is built...")

saver = tf.train.Saver(max_to_keep=5)
init = tf.global_variables_initializer()

test_df_cate = test_df.loc[:,config.cate_cols]
test_df_cont = test_df.loc[:,config.cont_cols]

test_step = math.ceil(len(test_df)/ config.batch_size)

# test
print("Predicting...")
with tf.Session() as sess:
    saver.restore(sess, '.\\ckpt\\best1.ckpt')
    preds_cate_, preds_cont_ = run_session(sess, test_step, [test_df_cate, test_df_cont], config, mdl, mode=4)

    preds_cate_1 = np.concatenate(preds_cate_)
    preds_cont_1 = np.concatenate(preds_cont_)

    saver.restore(sess, '.\\ckpt\\best2.ckpt')
    preds_cate_, preds_cont_ = run_session(sess, test_step, [test_df_cate, test_df_cont], config, mdl, mode=4)

    preds_cate_2 = np.concatenate(preds_cate_)
    preds_cont_2 = np.concatenate(preds_cont_)

    saver.restore(sess, '.\\ckpt\\best3.ckpt')
    preds_cate_, preds_cont_ = run_session(sess, test_step, [test_df_cate, test_df_cont], config, mdl, mode=4)

    preds_cate_3 = np.concatenate(preds_cate_)
    preds_cont_3 = np.concatenate(preds_cont_)

preds_cate_ = (preds_cate_1 + preds_cate_2 + preds_cate_3)/3
preds_cont_ = (preds_cont_1 + preds_cont_2 + preds_cont_3)/3


#out
print("Exproting...")
#cliping
preds_cont_[preds_cont_<0] = 0.0

# Categorical Vals Restore
pred_args = []
for p_ in preds_cate_:
    start_idx = 0
    pred_arg = []
    for kl in config.cate_lens:
        pred_arg.append(np.argmax(p_[start_idx: start_idx+kl]))
        start_idx += kl
    pred_args.append(pred_arg)
pred_args_np = np.array(pred_args)   

# Merge
test_df[config.cate_cols] = config.test_cate_mask * pred_args_np + test_df[config.cate_cols].values * ((config.test_cate_mask - 1)*(-1))
test_df[config.cont_cols] = config.test_cont_mask * preds_cont_  + test_df[config.cont_cols].values * ((config.test_cont_mask - 1)*(-1))

for key in config.cate_cols:
    test_df.loc[:,key] = test_df.loc[:,key].apply(lambda x: [k for k, v in config.kv_map[key].items() if v == x][0])

# Export
os.makedirs(config.dir_test, exist_ok=True)
test_df.to_csv(config.path_test, encoding='cp949', index=False, columns=config.test_cols)
print("Output_File is Exported to {}".format(os.path.abspath(config.path_test)))

df_test_out = pd.read_csv(config.path_test, encoding='cp949')
df_result = pd.read_csv('./data/result_kor.csv', encoding='cp949')

result_list = []
for r, c in zip(df_result.행, df_result.열):
    result_list.append(df_test_out.iloc[r-2][config.cols_dict[c]])

df_result.값 = result_list
df_result.to_csv(config.path_result, encoding='cp949', index=False)
print("Result_File is Exported to {}".format(os.path.abspath(config.path_result)))
#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import gc

import math
import numpy as np
import pandas as pd
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import xgboost as xgb

from datetime import datetime


train = pd.read_csv('raw_data/train.csv')
train = train.sort_values(by=['QUEUE_ID', 'DOTTING_TIME']).reset_index(drop=True)

test = pd.read_csv('raw_data/evaluation_public.csv')
test = test.sort_values(by=['ID', 'DOTTING_TIME']).reset_index(drop=True)

sub_sample = pd.read_csv('raw_data/submit_example.csv')


train['DISK_USAGE'].fillna(0, inplace=True)
train = train.dropna()

train = train[train.STATUS=='available']
train = train[train.PLATFORM=='x86_64']
train = train[train.RESOURCE_TYPE=='vm']

del train['STATUS']
del train['PLATFORM']
del train['RESOURCE_TYPE']

del test['STATUS']
del test['PLATFORM']
del test['RESOURCE_TYPE']

del train['FAILED_JOB_NUMS']
del train['CANCELLED_JOB_NUMS']

del test['FAILED_JOB_NUMS']
del test['CANCELLED_JOB_NUMS']

def timework(t):
    return t.hour*60 + t.minute


train['DOTTING_TIME'] = pd.to_datetime(train['DOTTING_TIME'],unit='ms')
train['DOTTING_TIME'] = train['DOTTING_TIME'].apply(timework)

test['DOTTING_TIME'] = pd.to_datetime(test['DOTTING_TIME'],unit='ms')
test['DOTTING_TIME'] = test['DOTTING_TIME'].apply(timework)

le = LabelEncoder()
train['QUEUE_TYPE'] = le.fit_transform(train['QUEUE_TYPE'].astype(str))
test['QUEUE_TYPE'] = le.transform(test['QUEUE_TYPE'].astype(str))


# t0 t1 t2 t3 t4  ->  t5 t6 t7 t8 t9 
# t1 t2 t3 t4 t5  ->  t6 t7 t8 t9 t10

df_train = pd.DataFrame()
threshold = 15
for id_ in tqdm(train.QUEUE_ID.unique()):
    df_tmp = train[train.QUEUE_ID == id_]
    features = list()
    t_cpu = list()
    values = df_tmp.values
    for i, _ in enumerate(values):
        if i + 10 < len(values):
            li_v = list()
            li_v.append(values[i][0])
            li_cpu = list()
            flag = False
            for j in range(5):
                li_v.extend(values[i+j][3:].tolist())
                li_cpu.append(values[i+j+5][3])
            if flag:
                continue
            features.append(li_v)
            t_cpu.append(li_cpu)
    df_feat = pd.DataFrame(features)
    df_feat.columns = ['QUEUE_ID',
                       'CPU_USAGE_1', 'MEM_USAGE_1',
                       'LAUNCHING_JOB_NUMS_1', 'RUNNING_JOB_NUMS_1',
                       'SUCCEED_JOB_NUMS_1', 'DOTTING_TIME1', 'DISK_USAGE_1',
                       'CPU_USAGE_2', 'MEM_USAGE_2',
                       'LAUNCHING_JOB_NUMS_2', 'RUNNING_JOB_NUMS_2',
                       'SUCCEED_JOB_NUMS_2', 'DOTTING_TIME2', 'DISK_USAGE_2',
                       'CPU_USAGE_3', 'MEM_USAGE_3',
                       'LAUNCHING_JOB_NUMS_3', 'RUNNING_JOB_NUMS_3',
                       'SUCCEED_JOB_NUMS_3', 'DOTTING_TIME3', 'DISK_USAGE_3',
                       'CPU_USAGE_4', 'MEM_USAGE_4',
                       'LAUNCHING_JOB_NUMS_4', 'RUNNING_JOB_NUMS_4',
                       'SUCCEED_JOB_NUMS_4', 'DOTTING_TIME4', 'DISK_USAGE_4',
                       'CPU_USAGE_5', 'MEM_USAGE_5',
                       'LAUNCHING_JOB_NUMS_5', 'RUNNING_JOB_NUMS_5',
                       'SUCCEED_JOB_NUMS_5', 'DOTTING_TIME5', 'DISK_USAGE_5',
                      ]
    df_cpu = pd.DataFrame(t_cpu)
    df_cpu.columns = ['cpu_1', 'cpu_2', 'cpu_3', 'cpu_4', 'cpu_5']
    df = pd.concat([df_feat, df_cpu], axis=1)
    print(f'QUEUE_ID: {id_}, lines: {df.shape[0]}')
    df_train = df_train.append(df)


df_test = pd.DataFrame()

for id_ in tqdm(test.QUEUE_ID.unique()):
    df_tmp = test[test.QUEUE_ID == id_]
    features = list()
    values = df_tmp.values
    for i, _ in enumerate(values):
        if i % 5 == 0:
            li_v = list()
            li_v.append(values[i][0])
            li_v.append(values[i][1])
            for j in range(5):
                li_v.extend(values[i+j][4:].tolist())
            features.append(li_v)
    df_feat = pd.DataFrame(features)
    df_feat.columns = ['ID','QUEUE_ID',
                       'CPU_USAGE_1', 'MEM_USAGE_1',
                       'LAUNCHING_JOB_NUMS_1', 'RUNNING_JOB_NUMS_1',
                       'SUCCEED_JOB_NUMS_1', 'DOTTING_TIME1', 'DISK_USAGE_1',
                       'CPU_USAGE_2', 'MEM_USAGE_2',
                       'LAUNCHING_JOB_NUMS_2', 'RUNNING_JOB_NUMS_2',
                       'SUCCEED_JOB_NUMS_2', 'DOTTING_TIME2', 'DISK_USAGE_2',
                       'CPU_USAGE_3', 'MEM_USAGE_3',
                       'LAUNCHING_JOB_NUMS_3', 'RUNNING_JOB_NUMS_3',
                       'SUCCEED_JOB_NUMS_3', 'DOTTING_TIME3', 'DISK_USAGE_3',
                       'CPU_USAGE_4', 'MEM_USAGE_4',
                       'LAUNCHING_JOB_NUMS_4', 'RUNNING_JOB_NUMS_4',
                       'SUCCEED_JOB_NUMS_4', 'DOTTING_TIME4', 'DISK_USAGE_4',
                       'CPU_USAGE_5', 'MEM_USAGE_5',
                       'LAUNCHING_JOB_NUMS_5', 'RUNNING_JOB_NUMS_5',
                       'SUCCEED_JOB_NUMS_5', 'DOTTING_TIME5', 'DISK_USAGE_5',
                      ]
    df = df_feat.copy()
    print(f'QUEUE_ID: {id_}, lines: {df.shape[0]}')
    df_test = df_test.append(df)




def run_lgb_qid(df_train, df_test, target, qid):
    feature_names = list(
        filter(lambda x: x not in ['QUEUE_ID', 'CU', 'QUEUE_TYPE'] + [f'cpu_{i}' for i in range(1, 6)],
               df_train.columns))

    # 提取 QUEUE_ID 对应的数据集
    df_train = df_train[df_train.QUEUE_ID == qid]
    df_test = df_test[df_test.QUEUE_ID == qid]

    print(f"QUEUE_ID:{qid}, target:{target}, train:{len(df_train)}, test:{len(df_test)}")

    model = lgb.LGBMRegressor(num_leaves=80,
                              max_depth=7,
                              learning_rate=0.05,
                              n_estimators=10000,
                              subsample=0.9,
                              feature_fraction=0.8,
                              reg_alpha=0.5,
                              reg_lambda=0.8,
                              random_state=2020,
                              )
    oof = []
    prediction = df_test[['ID', 'QUEUE_ID']]
    prediction[target] = 0

    kfold = KFold(n_splits=20, random_state=42)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric='mse',
                              #eval_metric = 'mae',
                              early_stopping_rounds=40)

        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][[target, 'QUEUE_ID']].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict(df_test[feature_names], num_iteration=lgb_model.best_iteration_)
        prediction[target] += pred_test / kfold.n_splits

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    df_oof = pd.concat(oof)
    '''
    score = mean_squared_error(df_oof[target], df_oof['pred'])
    print('MSE:', score)
    '''
    score = mean_absolute_error(df_oof[target], df_oof['pred'])/100
    print('mae:', score)

    return prediction, score


predictions = list()
scores = list()
qid_scores = list()

for qid in tqdm(test.QUEUE_ID.unique()):    
    df = pd.DataFrame()
    index = 0
    sum_score = 0
    for t in [f'cpu_{i}' for i in range(1,6)]:
        prediction, score = run_lgb_qid(df_train, df_test, target=t, qid=qid)
        if t == 'cpu_1':
            df = prediction.copy()
        else:
            df = pd.merge(df, prediction, on=['ID', 'QUEUE_ID'], how='left')
            sum_score += score
        index += 1
    scores.append(sum_score)
    qid_scores.append((qid,sum_score))

    predictions.append(df)



sum_mae1=0.0
sum_mae2=0.0
sum_mae3=0.0
sum_sample=0
for tu in qid_scores:
    sum_mae1 += train[train.QUEUE_ID == tu[0]].shape[0] * tu[1]
    sum_mae2 += tu[1]
    sum_mae3 += test[test.QUEUE_ID == tu[0]].shape[0] * tu[1]
    sum_sample += train[train.QUEUE_ID == tu[0]].shape[0]

#总的mae，三种不同计算方式
print('MAE1: ',sum_mae1 / sum_sample)
print('MAE2: ',sum_mae2 / len(qid_scores))
print('MAE3: ',sum_mae3 / test.shape[0])
for tu in qid_scores: #每个队列的mae
    print(tu)

sub = pd.concat(predictions)



sub = sub.sort_values(by='ID').reset_index(drop=True)
sub.drop(['QUEUE_ID'], axis=1, inplace=True)
sub.columns = ['ID'] + [f'CPU_USAGE_{i}' for i in range(1,6)]


avg = pd.read_csv('raw_data/avg.csv')


def work(x):

    if  x<=20:
        return x
    else:
        return x + 0.6


for col in [f'CPU_USAGE_{i}' for i in range(1,6)]:
    sub[col] = sub[col].apply(work)
    #sub2[col] = sub2[col].apply(work)


for col in [f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]:
    sub[col] = avg[col]
    #sub2[col] = avg[col]

sub = sub[['ID',
           'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1', 
           'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2', 
           'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3', 
           'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4', 
           'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']]


# 注意: 提交要求预测结果需为非负整数, 包括 ID 也需要是整数

sub['ID'] = sub['ID'].astype(int)
#sub2['ID'] = sub2['ID'].astype(int)


for col in [i for i in sub.columns if i != 'ID']:
    sub[col] = sub[col].apply(np.floor)
    sub[col] = sub[col].apply(lambda x: 0 if x<0 else x)
    sub[col] = sub[col].apply(lambda x: 100 if x>100 else x)
    sub[col] = sub[col].astype(int)
    
for col in [f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]:
    sub[col] = avg[col]
    #sub2[col] = avg[col]
sub.to_csv('result12030833.csv', index=False)

print(datetime.now())



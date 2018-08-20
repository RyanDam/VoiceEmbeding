import os
import numpy as np

import h5py
import matplotlib.pyplot as plt

import random
from random import shuffle

from tqdm import tqdm

def stat(arr):
    arr = np.array(arr)
    print((arr.shape, 'min:', np.min(arr), 'max:', np.max(arr), 'std:', np.std(arr), 'mean:', np.mean(arr), 'median:', np.median(arr)))

def get_weight_dict(numpy_train_x, numpy_train_y):
    stat_dic, stat_key, stat_num = statistic_data(numpy_train_x, numpy_train_y)
    maxx = np.max(stat_num)
    stat_weights = (maxx/stat_num).astype(np.uint8)
    stat_weights = stat_weights*10/np.sum(stat_weights)

    weights_dict = {}
    for k, w in zip(stat_key, stat_weights):
        weights_dict[k] = w
        
    return weights_dict

def divide_data(numpy_train_x, numpy_train_y, chunk_num=5, keeps=[]):
    
    assert chunk_num > 2, 'Chunnk size must greater than 2'
    
    def get_sample(arr, chunk_num):
        shuffle(arr)
        
        percent = float(1/chunk_num)
        size = int(percent*len(arr))
        
        f = arr[:size]
        if chunk_num == 3:
            s = arr[size:size+size]
            l = arr[size + size:]
            return [f, s, l]
        
        res = [f]
        for i in range(1,chunk_num - 1):
            res.append(arr[size*i:size*i + size])
        
        res.append(arr[(chunk_num - 1)*size:])
        
        return res
    
    stat_dic, stat_key, stat_num = statistic_data(numpy_train_x, numpy_train_y)
    
    res = []
    for i in range(0, chunk_num):
        res.append({})
        
    for k in stat_dic:
        if k in keeps:
            for d in res:
                d[k] = stat_dic[k]
        else:
            samples = get_sample(stat_dic[k], chunk_num)
            for d, s in zip(res, samples):
                d[k] = s
    
    return res

def statistic_data(numpy_train_x, numpy_train_y, title=None):
    data_dic = {}
    for y, p in zip(numpy_train_y, numpy_train_x):
        if y in data_dic:
            data_dic[y] = data_dic[y] + [p]
        else:
            data_dic[y] = [p]

    keys = list(data_dic.keys())
    nums = [len(data_dic[k]) for k in keys]
    
    if title:
        print(title, 'sum:', np.sum(nums))
        plt.bar(keys, nums)
        plt.show()
    
    print(nums)
    print(keys)
    stat(nums)
    
    return data_dic, keys, nums

def upsample_data(numpy_train_x, numpy_train_y):
    data_dic, keys, nums = statistic_data(numpy_train_x, numpy_train_y, title='before')

    nums_arr = np.array(nums)
    maxx = np.max(nums_arr)
    scalenum = (maxx/nums_arr + 0.5).astype(np.uint8)

    for key, scale in zip(keys, scalenum):
        data_dic[key] = (data_dic[key]*scale)[:maxx]

    numdata11, numdata22 = convert_dict_to_pair(data_dic)

    ddata_dic, keys, nums = statistic_data(numdata11, numdata22, title='after')
    
    return numdata11, numdata22

def convert_dict_to_pair(data_dic):
    numdata1 = []
    numdata2 = []
    for k in data_dic:
        numdata1.extend(data_dic[k])
        numdata2.extend([k]*len(data_dic[k]))

    pair = [(a, b) for a, b in zip(numdata1, numdata2)]
    shuffle(pair)
    numdata11 = [a[0] for a in pair]
    numdata22 = [a[1] for a in pair]
    
    return numdata11, numdata22

def split_data(xs, ys, alpha=0.9, maxx=None):
    data_dic, keys, nums = statistic_data(xs, ys)
    
    def get_sample(arr, alpha=0.9, maxx=None):
        shuffle(arr)
        i = int(alpha*len(arr))
        if maxx:
            if alpha <= 0.5:
                if i > maxx:
                    i=maxx
            else:
                if len(arr) - i > maxx:
                    i = len(arr) - maxx
        return arr[:i], arr[i:]
    
    sdict_1, sdict_2 = {}, {}
    for k in data_dic:
        a1, a2 = get_sample(data_dic[k], alpha, maxx)
        sdict_1[k] = a1
        sdict_2[k] = a2
    
    x1, y1 = convert_dict_to_pair(sdict_1)
    x2, y2 = convert_dict_to_pair(sdict_2)
    return x1, y1, x2, y2

def split_data_2(xs, ys, alpha=1000):
    data_dic, keys, nums = statistic_data(xs, ys)
    
    def get_sample(arr, alpha=1000):
        shuffle(arr)
        i = alpha
        return arr[:i], arr[i:]
    
    sdict_1, sdict_2 = {}, {}
    for k in data_dic:
        a1, a2 = get_sample(data_dic[k], alpha)
        sdict_1[k] = a1
        sdict_2[k] = a2
    
    x1, y1 = convert_dict_to_pair(sdict_1)
    x2, y2 = convert_dict_to_pair(sdict_2)
    return x1, y1, x2, y2
    
def get_wave_pair(data_path):
    
    numpy_test_x = []
    numpy_test_y = []

    for a,b,c in os.walk(data_path):
        for fname in c:
            numpy_test_x.append(os.path.join(data_path, fname))
            label_indice = int(fname.split('_')[0])
            numpy_test_y.append(label_indice)
        break
    
    return numpy_test_x, numpy_test_y

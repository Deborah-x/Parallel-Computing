import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def train_proc(spath:str) -> None:
    """
    Data processing for training set
    :param spath: the path of the training set with the suffix '.json' 
    """
    # Load a dataset with the suffix '.json'
    with open(spath, 'r') as f:
        dataset = json.load(f)  # type 'list'

    # Convert the training set to type 'dataframe' and put them into a file named dpath
    dpath = 'train_proc.txt'
    pd.DataFrame(dataset).to_csv(dpath, index=False)

    # Discard the data with empty 'fit' index in the training set and put the remaining data into the old file
    data = pd.read_csv(dpath)
    data.dropna(subset=['fit']).to_csv(dpath, index=False)
    

def test_proc(spath:str) -> None:
    """
    Data processing for testing set
    :param spath: the path of the testing set with the suffix '.json' 
    """
    # Load a dataset with the suffix '.json'
    with open(spath, 'r') as f:
        dataset = json.load(f)  # type 'list'

    # Convert the training set to type 'dataframe' and put them into a file named dpath
    dpath = 'test_proc.txt'
    pd.DataFrame(dataset).to_csv(dpath, index=False)

    # If the test set has no empty 'fit' index, this is useless
    # Discard the data with empty 'fit' index in the training set and put the remaining data into the old file
    data = pd.read_csv(dpath)
    data.dropna(subset=['fit']).to_csv(dpath, index=False)

def val_range(path:str, key:str) -> set:
    '''
    Return the range of keywords in a file
    :param path: the path of the file we want to search
    :param key: keywords
    '''
    df = pd.read_csv(path)
    return set(df[key])

def build_dict() -> dict:
    '''
    Create a dictionary to store an optional set for each indicator
    '''
    path = 'train_proc.txt'
    df = pd.read_csv(path)
    item_name_list = list(set(df['item_name']))
    rented_for_list = list(set(df['rented_for']))
    usually_wear_list = list(set(df['usually_wear']))
    size_list = list(set(df['size']))
    age_list = list(set(df['age']))
    height_list = list(set(df['height']))
    bust_size_list = list(set(df['bust_size']))
    weight_list = list(set(df['weight']))
    body_type_list = list(set(df['body_type']))
    price_list = list(set(df['price']))
    dic = {'item_name':item_name_list, 'rented_for':rented_for_list, 'usually_wear':usually_wear_list, 'size':size_list, 'age':age_list, 'height':height_list, 'bust_size':bust_size_list, 'weight':weight_list, 'body_type':body_type_list, 'price':price_list}
    return dic

def data2vector(spath:str) -> np.ndarray:
    '''
    Convert each piece of data into a vector of length 10, and save the converted array in the file 'data2vector.npy' for convenient further use
    '''
    dic = build_dict()
    dataset = pd.read_csv(spath)
    m, n = dataset.shape
    feature = ['item_name', 'rented_for', 'usually_wear', 'size', 'age', 'height', 'bust_size', 'weight', 'body_type', 'price']
    X = np.zeros((m, 10))
    for i in tqdm(range(m)):
        x = dataset.iloc[i]
        for j in feature:
            l = dic[j]
            if x[j] in l:
                X[i, feature.index(j)] = l.index(x[j])
    return X

def label2vector() -> np.ndarray:
    '''
    Convert three categories into number 1,2,3, and save the converted array in the file 'label2vector.npy' for convenient further use
    '''
    path ='train_proc.txt'
    data = pd.read_csv(path)
    label = data['fit']
    label = label.replace(['Small', 'True to Size', 'Large'], [1, 2, 3]).to_numpy()
    return label
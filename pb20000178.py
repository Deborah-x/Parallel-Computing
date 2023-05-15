import json
import pandas as pd
import numpy as np
from kNN import *
from data_proc import *
from sklearn.metrics import f1_score

DATA_PATH = "./train_data_all.json"

## for TA's test
## you need to modify the class name to your student id.
## you also need to implement the predict function, which reads the .json file,
## calls your trained model and returns predict results as an ndarray

class PB20000178():
    def predict(self, data_path): 
        # a predicting system
        pred = []
        print('-----------------Training-----------------')
        balance_method()
        print('------------Processing Testset------------')
        data_test = TestSet(data_path)
        m, _ = data_test.shape
        Store_data = np.load('Store_data.npy')
        Store_label = np.load('Store_label.npy')
        print('-----------------Testing------------------')
        for i in tqdm(range(m)):
            r = classify(data_test[i], Store_data[:], Store_label[:], 5)
            pred.append(r)
        return np.array(pred)


## for local validation
if __name__ == '__main__':
    with open(DATA_PATH, "r") as f:
        test_data_list = json.load(f)
    true = np.array([int(data["fit"]) for data in test_data_list])
    bot = PB20000178()
    pred = bot.predict(DATA_PATH)
    # print(pred.shape)

    macro_f1 = f1_score(y_true=true, y_pred=pred, average="macro")
    f_1 = f1_score(y_true=(true==1), y_pred=(pred==1))
    f_2 = f1_score(y_true=(true==2), y_pred=(pred==2))
    f_3 = f1_score(y_true=(true==3), y_pred=(pred==3))
    # For multiple classification tasks，sklearn calculates macro_f1 by calculating f1 values for each classification and then calculating the mean value
    print('macro_f1 = ', macro_f1)
    print('\nf_1 = ', f_1)
    print('f_2 = ', f_2)
    print('f_3 = ', f_3)
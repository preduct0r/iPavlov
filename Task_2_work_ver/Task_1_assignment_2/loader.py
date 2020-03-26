import dill
import os
from varname import varname


data_path = r'C:\Users\Andrey\Google Диск\courses\DeepPavlov\Task-2-preduct0r\data\task3'

def saver(data, data_name):
    with open(os.path.join(data_path, data_name), "wb")as f:
        dill.dump(data, f)

def loader(data_name):
    with open(os.path.join(data_path, data_name), "rb")as f:
        return dill.load(f)


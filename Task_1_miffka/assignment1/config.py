import os.path as osp

CURRENT_PATH = osp.dirname(osp.realpath(__file__))

class Config():
    root_dir = osp.join(CURRENT_PATH, "..")
    data_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "data"))
    model_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "models"))
    logs_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "runs"))
    
config = Config()

import json
import pickle
import numpy as np
from datasets.dataloader import *

gen_number = 100
num_classes = 100
config = json.load(open("./configs/config_base.json"))

trainset, testset = cifarOriginal(config['data_dir'], [], [])

t = np.array(testset.targets)

support = {}

support['indices'] = [[np.random.permutation(np.argwhere(t == i))[j][0] for j in range(gen_number)] for i in range(num_classes)]
support['images'] = [[testset.data[np.random.permutation(np.argwhere(t == i))[j][0]] for j in range(gen_number)] for i in range(num_classes)]

with open(os.path.join(config['data_dir'], config['support_file']), "wb") as data_file:
    pickle.dump(support, data_file)
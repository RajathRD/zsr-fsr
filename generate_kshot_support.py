import json
import pickle
import numpy as np
from datasets.dataloader import *

config = json.load(open("config.json"))

trainset, testset = cifarOriginal(config['data_dir'], [], [])

t = np.array(testset.targets)

support = {}

support['indices'] = [[np.random.permutation(np.argwhere(t == i))[j][0] for j in range(5)] for i in range(100)]
support['images'] = [[testset.data[np.random.permutation(np.argwhere(t == i))[j][0]] for j in range(5)] for i in range(100)]

with open(os.path.join(config['data_dir'], "cifar_kshot_support.pkl"), "wb") as data_file:
    pickle.dump(support, data_file)
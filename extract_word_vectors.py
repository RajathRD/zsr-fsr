import os
import numpy as np
import io
import fasttext
import fasttext.util
import pickle
import json
import chakin
from sklearn.preprocessing import normalize

from gensim import models
from datasets.dataloader import cifarOriginal

config = json.load(open("config.json"))

data_dir = config['w2v_dir']

print (config['data_dir'], data_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

words = cifarOriginal(config['data_dir'], [], [])[0].classes

def normalize(x):
    return x/np.sqrt(np.sum(np.square(x)))

def save_vectors(file_name, w2v):
    pickle.dump(w2v, open(f'{file_name}.pkl', 'wb'))
    print (w2v['apple'])
    for w in w2v:
        w2v = normalize(w2v[w])

    pickle.dump(w2v, open(f'{file_name}_unitnorm.pkl', 'wb'))

def extract_w2v(words, vocab2vec=None, get_vector_fn=None):
    w2v = {}
    for w in words:
        ws = w.split("_")
        if get_vector_fn is not None:
            vector = np.sum(np.vstack([get_vector_fn(wix) for wix in ws]), axis=0)
        elif w2v is not None:
            vector = np.sum(np.vstack([vocab2vec[wix] for wix in ws]), axis=0)
        w2v[w] = vector
        # print (w, vector, len(ws))
    return w2v

def fasttext_(file_name='cc.en.300.bin'):
    # downladed with fastext util given below
    # refer: https://fasttext.cc/docs/en/crawl-vectors.html
    global words
    path = os.path.join(data_dir, file_name)
    print('Downloading fastext if not available')
    if not os.path.exists(path):
        fasttext.util.download_model('en')

    print ("Loading fasttext vectors...")
    ft = fasttext.load_model(path)

    return extract_w2v(words, get_vector_fn=ft.get_word_vector)

def glove(file_name='glove.42B.300d.txt'):
    # downloaded file using chakin
    # after download make sure to extract the file
    w2v = {}
    path = os.path.join(data_dir, file_name)
    if "twitter" not in path and not os.path.exists(path):
        chakin.download(number=15, save_dir=data_dir)

    elif "twitter" in path and not os.path.exists(path):
        chakin.download(number=20, save_dir=data_dir)

    print (f"Loading GloVe ({file_name})...")
    with open(path, 'r') as data_file:
        for line in data_file:
            line_s = line.split()
            w = line_s[0]
            vector = np.array(line_s[1:], dtype=np.float64)
            w2v[w] = vector
    
    return extract_w2v(words, vocab2vec=w2v)

def google_news(file_name='GoogleNews-vectors-negative300.bin'):
    # downloaded file using chakin
    # after download make sure to extract the file
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        chakin.download(number=21, save_dir=data_dir)
    print ("Loading Google News vectors")
    w2v = models.KeyedVectors.load_word2vec_format(path, binary=True)

    return extract_w2v(words, vocab2vec=w2v)
    

if __name__ == "__main__":
    save_vectors("w2v_fasttext", fasttext_())
    save_vectors("w2v_glove", glove())
    save_vectors("w2v_google", google_news())

    # we don't use glove twitter because it doesn't contian the word flatfish
    # save_vectors("w2v_glove.pkl", glove('glove.twitter.27B.200d.txt'))
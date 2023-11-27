import pandas as pd
import numpy as np
import pickle
import scipy
import gc
from sklearn.decomposition import PCA, TruncatedSVD,KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from umap import UMAP
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numba # for speeding up

train = pd.read_hdf("/root/autodl-tmp/open-problems-multimodal/train_cite_inputs.h5")
test = pd.read_hdf("/root/autodl-tmp/open-problems-multimodal/test_cite_inputs.h5")
train_columns = train.columns
train_indexes = train.index
print(train.shape)
all_zeros_features = train.columns[train.sum()==0].to_list()
none_zeros_features = [i for i in train.columns if i not in all_zeros_features]
len(all_zeros_features)
test_indexes = test.index
train = train[none_zeros_features]
test = test[none_zeros_features]
print(test.shape)
train = scipy.sparse.csr_matrix(train)
test = scipy.sparse.csr_matrix(test)
all = scipy.sparse.vstack([train,test])
del train,test
all_indexes = train_indexes.to_list()+test_indexes.to_list()
all_log = np.log1p(all)

def get_tsvd():
    all_log = np.log1p(all)
    all_indexes = train_indexes.to_list()+test_indexes.to_list()
    pure_tsvd = TruncatedSVD(n_components=128, random_state=42)
    train_tsvd = pure_tsvd.fit_transform(all_log)
    print(pure_tsvd.explained_variance_ratio_.sum())
    train_tsvd = pd.DataFrame(train_tsvd,index = all_indexes)
    test = train_tsvd.iloc[70988:]
    test = test.drop_duplicates()
    # test = test.reindex(test_indexes_ori)
    test = test.fillna(0)
    test.shape
    np.savez("/root/autodl-tmp/feature/cite_train_tsvd.npz", train_tsvd.iloc[:70988])
    np.savez("/root/autodl-tmp/feature/cite_test_tsvd.npz",test)
    del train_tsvd,pure_tsvd,test
    gc.collect()
    
def get_umap():
    umap = UMAP(n_neighbors = 16,n_components=128, random_state=42,verbose = True,low_memory = True,n_jobs = -1)
    train_umap = umap.fit_transform(all_log.toarray())
    train_umap = pd.DataFrame(train_umap,index = all_indexes)
    test = train_umap.iloc[70988:]
    test = test.drop_duplicates()
    # test = test.reindex(test_indexes_ori)
    test = test.fillna(0)
    test.shape
    np.savez("/root/autodl-tmp/feature/cite_train_umap.npz", train_umap.iloc[:70988])
    np.savez("/root/autodl-tmp/feature/cite_test_umap.npz",test)
    del train_umap,umap,test
    gc.collect()
    
def tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf
def get_novel():
    all_novel = tfidf(all)
    all_novel = all_novel.tocsr()
    all_novel = np.log1p(all_novel * 1e4)
    tsvd = TruncatedSVD(n_components=128, random_state=42)
    train_novel = tsvd.fit_transform(all_novel)
    print(tsvd.explained_variance_ratio_.sum())
    train_novel = pd.DataFrame(train_novel,index = all_indexes)
    test = train_novel.iloc[70988:]
    test = test.drop_duplicates()
    # test = test.reindex(test_indexes_ori)
    test = test.fillna(0)
    test.shape
    np.savez("/root/autodl-tmp/feature/cite_train_novel.npz", train_novel.iloc[:70988])
    np.savez("/root/autodl-tmp/feature/cite_test_novel.npz",test)
    del train_novel,tsvd,test
    gc.collect()
def get_all():
    train_tsvd = np.load("/root/autodl-tmp/feature/cite_train_tsvd.npz")["arr_0"]
    train_umap = np.load("/root/autodl-tmp/feature/cite_train_umap.npz")["arr_0"]
    train_novel = np.load("/root/autodl-tmp/feature/cite_train_novel.npz")["arr_0"]
    train_all  = np.concatenate([train_tsvd, train_umap, train_novel],axis = 1)
    test_tsvd = np.load("/root/autodl-tmp/feature/cite_test_tsvd.npz")["arr_0"]
    test_umap = np.load("/root/autodl-tmp/feature/cite_test_umap.npz")["arr_0"]
    test_novel = np.load("/root/autodl-tmp/feature/cite_test_novel.npz")["arr_0"]

    test_all  = np.concatenate([test_tsvd, test_umap, test_novel],axis = 1)
    np.savez("/root/autodl-tmp/feature/new_cite_train_final.npz", train_all)
    np.savez("/root/autodl-tmp/feature/new_cite_test_final.npz",test_all)
    
get_tsvd()
print("get_tsvd finished")
get_umap()
print("get_umap finished")
get_novel()
print("get_novel finished")
get_all()
print("get_all finished")


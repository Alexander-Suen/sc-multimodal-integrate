import sys
import time
import gc
import glob
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder
from net import CNN,ori_Tester
from torch import nn
import seaborn as sns
from loss import correl_loss
from trainer import Cite_Trainer

train = np.load("/root/autodl-tmp/feature/new_cite_train_final.npz")["arr_0"]
target = pd.read_hdf("/root/autodl-tmp/open-problems-multimodal/train_cite_targets.h5").values
print(train.shape,target.shape)
config = dict(
    atte_dims = 128,
    output_num = target.shape[1],
    input_num = train.shape[1],
    dropout = 0.1,
    mlp_dims = [train.shape[1]*2,train.shape[1]],
    
    layers = 5,
    patience = 5,
    max_epochs = 100,
    criterion = correl_loss,
    batch_size = 512,

    n_folds = 3,
    folds_to_train = [0,1,2],
    kfold_random_state = 42,

    tb_dir = "./log/",

    optimizer = torch.optim.AdamW,
    optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR,

    schedulerparams = dict(milestones=[6,10,15,20,25,30], gamma=0.1,verbose  = True), #9,12,15,20,25,30
    min_epoch = 11,
)

train_index = np.load("/root/code/train_cite_inputs_idxcol.npz",allow_pickle=True)
meta = pd.read_csv("/root/autodl-tmp/open-problems-multimodal/metadata.csv",index_col = "cell_id")
meta = meta[meta.technology=="citeseq"]
lbe = LabelEncoder()
meta["cell_type"] = lbe.fit_transform(meta["cell_type"])
meta["gender"] = meta.apply(lambda x:0 if x["donor"]==13176 else 1,axis =1)
meta_train = meta.reindex(train_index["index"])
train_meta = meta_train["gender"].values.reshape(-1, 1)
train = np.concatenate([train,train_meta],axis= -1)
train_meta = meta_train["cell_type"].values.reshape(-1, 1)
train = np.concatenate([train,train_meta],axis= -1)

def train_Loop():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"machine has {torch.cuda.device_count()} cuda devices")
        print(f"model of first cuda device is {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
    
    trainer = Cite_Trainer(device)
    kfold = GroupKFold(n_splits=config["n_folds"]) # , shuffle=True, random_state=config["kfold_random_state"]
    groups = np.random.randint(0, 5, train.shape[0])
    FOLDS_LIST = list(kfold.split(range(train.shape[0]),groups= groups)) #
    print("Training started")
    fold_scores = []
    for num_fold in config["folds_to_train"]:
        model = CNN(config)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        # model = model.to(device)
        best_score = trainer.train_one_fold(num_fold,FOLDS_LIST,train,target,model,config)
        fold_scores.append(best_score)
    print("\n")
    print(f"Final average score is {sum(fold_scores)/len(fold_scores)}")
    return fold_scores

def submit(test_pred,multi_path):
    submission = pd.read_csv(multi_path,index_col = 0)
    submission = submission["target"]
    print("data loaded")
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    assert not submission.isna().any()
    # submission = submission.round(6) # reduce the size of the csv
    print("start -> submission.zip")
    submission.to_csv('submission.csv')
    print("submission.zip saved!")
    
def test(fold_scores):
    test = np.load("/root/autodl-tmp/feature/new_cite_test_final.npz")["arr_0"]

    test_index = np.load("/root/code/test_cite_inputs_idxcol.npz",allow_pickle=True)
    meta_test = meta.reindex(test_index["index"])
    test_meta = meta_test["gender"].values.reshape(-1, 1)
    test = np.concatenate([test,test_meta],axis= -1)
    test_meta = meta_test["cell_type"].values.reshape(-1, 1)
    test = np.concatenate([test,test_meta],axis= -1)
    tester = ori_Tester( torch.device("cuda:0"),config,fold_scores)
    test_pred = tester.test(test)
    submit(test_pred,"/root/autodl-tmp/open-problems-multimodal/sample_submission.csv")
    sns.heatmap(test_pred)
    plt.savefig('heatmap.png')
    
if __name__ == "__main__":
    fold_scores = train_Loop()
    test(fold_scores)
   



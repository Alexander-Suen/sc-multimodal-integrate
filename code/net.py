import sys
import time
from tqdm.notebook import tqdm
import gc
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder

class conv_block(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        mlp_dims = config["mlp_dims"]

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

    def forward(self,x):
        x1 = self.conv_2(x)
        x2 = self.conv_2_1(x)
        x3 = self.conv_2_2(x)
        x = x1+x2+x3+x
        return x 
class CNN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        mlp_dims = config["mlp_dims"]
        self.layers = config["layers"]

        self.backbone = torch.nn.Linear(self.input_num ,self.input_num)
        self.embedding_1 = torch.nn.Embedding(2,256)
        self.embedding_2 = torch.nn.Embedding(7,256)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.input_num,4096),
            # torch.nn.Linear(2048,4096)
        )
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            # torch.nn.Mish(),

            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

        self.conv_layers = torch.nn.ModuleList()
        for i in range(self.layers):
            self.conv_layers.append(conv_block(config))

        self.final = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4096,2048),
            torch.nn.Mish(),
            torch.nn.Linear(2048,512),
            torch.nn.Mish(),
            torch.nn.Linear(512,output_num),
            torch.nn.Mish(),
        )
    
    def forward(self,x):
        x_ = self.embedding_2(x[:,-1].int())
        x_ = torch.repeat_interleave(torch.unsqueeze(x_,-1),16,-1)
        x = self.proj(x[:,:self.input_num])
        x = torch.reshape(x,(x.shape[0],256,16))
        x = x+x_
        x1 = self.conv_1(x)
        x2 = self.conv_1_1(x)
        x3 = self.conv_1_2(x)
        # res_list = []
        x = x1+x2+x3
        # res_list.append(x)

        for layer in self.conv_layers:
            x = layer(x)
            # res_list.append(x)

        # x = torch.concat(res_list,dim =-1)
        x = self.final(x)
        return x
class ori_Tester:
    def __init__(self,device,config,weight):
        self.weight = weight
        self.device = device
        self.config = config

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)
        # return (x - np.mean(x)) / np.std(x)
    
    def test_fn_ensemble(self,model_list, dl_test):
        res = np.zeros(
            (self.len, self.config["output_num"]), )
        
        for model in model_list:
            model.eval()
            
        cur = 0
        for inpt in (dl_test):
            inpt = inpt[0]
            mb_size = inpt.shape[0]

            with torch.no_grad():
                pred_list = []
                inpt = inpt.to(self.device)
                # print("inpt",inpt.shape)
                for id,model in enumerate(model_list):
                    model.to(self.device)
                    model.eval()
                    pred = model(inpt)
                    model.to("cpu")
                    # print("pred",pred.shape)
                    pred = self.std(pred.cpu().numpy())* self.weight[id]
                    pred_list.append(pred)
                pred = sum(pred_list)/len(pred_list)
                
            # print(res.shape, cur, cur+pred.shape[0], res[cur:cur+pred.shape[0]].shape, pred.shape)
            res[cur:cur+pred.shape[0]] = pred
            cur += pred.shape[0]
                
        return {"preds":res}

    def load_model(self,path ):
        model_list = []
        for fn in (glob.glob(path)):
            prefix = fn[:-len("_best_params.pth")]
            config_fn = prefix + "_config.pkl"

            config = pickle.load(open(config_fn, "rb"))

            model = CNN(config)
            model.to("cpu")
            
            params = torch.load(fn)
            model.load_state_dict(params)
            
            model_list.append(model)
        print("model loaded")
        return model_list
    
    def load_data(self,test ):
        print("test inputs loaded")
        print(test.shape)
        self.len = test.shape[0]
        test = torch.tensor(test,dtype = torch.float)
        test = torch.utils.data.TensorDataset(test)
        return test

    def test(self,test,model_path = "./*_best_params.pth"):
        # self.weight = weight
        model_list = self.load_model(model_path)
        test_inputs = self.load_data(test)
        gc.collect()
        dl_test = torch.utils.data.DataLoader(test_inputs, batch_size=4096, shuffle=False, drop_last=False)
        test_pred = self.test_fn_ensemble(model_list, dl_test)["preds"]
        del model_list
        del dl_test
        del test_inputs
        gc.collect()
        print(test_pred.shape)
        # np.save("test_pred.npy",test_pred)
        return test_pred
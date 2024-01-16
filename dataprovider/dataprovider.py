import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader,Dataset


class AirQualityDataset(Dataset):
    def __init__(self, X, Y, masker):
        self.X = X
        self.Y = Y
        self.maskser = masker
        
    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        maskser = self.maskser[index]
        return X.astype(np.float32), Y.astype(np.float32), maskser.astype(np.bool_)

    def __len__(self):
        return len(self.X)


class AirQualityProvider:
    def __init__(self, input_len, batch_size, value):
        self.input_len = input_len
        self.batch_size = batch_size
        self.value = value
        self.preare_data()
        
    def preare_data(self,):
        self.data = pd.read_csv('./data/{}.csv'.format(self.value))
        self.masker = pd.read_csv('./data/{}_masker.csv'.format(self.value))
        self.tem = pd.read_csv('./data/match_TEM.csv')
        self.dpt = pd.read_csv('./data/match_DPT.csv')
        self.prs = pd.read_csv('./data/match_PRS.csv')
        self.rhu = pd.read_csv('./data/match_RHU.csv')
        self.win_d_inst = pd.read_csv('./data/match_WIN_D_INST.csv')
        self.win_s_inst = pd.read_csv('./data/match_WIN_S_INST.csv')
        
        with open('./data/train_validate_test_index.pckl', 'rb') as f:
            self.train_index, self.validate_index, self.test_index = pickle.load(f)
            
        self._min, self._max = np.load('./data/{}_min_max.npy'.format(self.value))
        self.tem_min, self.tem_max = np.load('./data/match_TEM_min_max.npy')
        self.dpt_min, self.dpt_max = np.load('./data/match_DPT_min_max.npy')
        self.prs_min, self.prs_max = np.load('./data/match_PRS_min_max.npy')
        self.rhu_min, self.rhu_max = np.load('./data/match_RHU_min_max.npy')
        self.win_d_inst_min, self.win_d_inst_max = np.load('./data/match_WIN_D_INST_min_max.npy')
        self.win_s_inst_min, self.win_s_inst_max = np.load('./data/match_WIN_S_INST_min_max.npy')
    
    def buildup_data(self,data_index):
        
        _dataset = []
        _tem_dataset = []
        _dpt_dataset = []
        _prs_dataset = []
        _rhu_dataset = []
        _win_d_inst_dataset = []
        _win_s_inst_dataset = []
        _masker = []
        _X = []
        _Y = []
        
        for idx in data_index:
            _masker.append(self.masker.iloc[idx, 5:].to_numpy())
            _dataset.append((self.data.iloc[idx, 5:].to_numpy() - self._min)/(self._max - self._min))
            _tem_dataset.append((self.tem.iloc[idx, 5:].to_numpy() - self.tem_min)/(self.tem_max- self.tem_min))
            _dpt_dataset.append((self.dpt.iloc[idx, 5:].to_numpy() - self.dpt_min)/(self.dpt_max- self.dpt_min))
            _prs_dataset.append((self.prs.iloc[idx, 5:].to_numpy() - self.prs_min)/(self.prs_max- self.prs_min))
            _rhu_dataset.append((self.rhu.iloc[idx, 5:].to_numpy() - self.rhu_min)/(self.rhu_max- self.rhu_min))
            _win_d_inst_dataset.append((self.win_d_inst.iloc[idx, 5:].to_numpy() - self.win_d_inst_min)/(self.win_d_inst_max-self.win_d_inst_min))
            _win_s_inst_dataset.append((self.win_s_inst.iloc[idx, 5:].to_numpy() - self.win_s_inst_min)/(self.win_s_inst_max - self.win_s_inst_min))
        _masker = np.stack(_masker, axis=0)
        _dataset = np.stack(_dataset, axis=0)
        _tem_dataset = np.stack(_tem_dataset, axis=0)
        _dpt_dataset = np.stack(_dpt_dataset, axis=0)
        _prs_dataset = np.stack(_prs_dataset, axis=0)
        _rhu_dataset = np.stack(_rhu_dataset, axis=0)
        _win_d_inst_dataset = np.stack(_win_d_inst_dataset, axis=0)
        _win_s_inst_dataset = np.stack(_win_s_inst_dataset, axis=0)
        
        _X.append(np.stack([_dataset[:, :self.input_len], 
            _tem_dataset[:, :self.input_len],
            _dpt_dataset[:, :self.input_len],
            _prs_dataset[:, :self.input_len],
            _rhu_dataset[:, :self.input_len],
            _win_d_inst_dataset[:, :self.input_len],
            _win_s_inst_dataset[:, :self.input_len],
            ], axis=-1))
        
        _Y.append(_dataset[:, self.input_len:])
        
        _masker = _masker[:, self.input_len:]
        
        _X = np.concatenate(_X)
        _Y = np.expand_dims(np.concatenate(_Y), axis=-1)
        
        return _X, _Y, _masker
        
    def train_loader(self,):
        train_X, train_Y, train_masker = self.buildup_data(self.train_index)
        train = AirQualityDataset(train_X, train_Y, train_masker )
        return DataLoader(dataset=train,batch_size=self.batch_size,shuffle=True,)
    
    def validate_loader(self,):
        validate_X, validate_Y, validate_masker  = self.buildup_data(self.validate_index)
        validate = AirQualityDataset(validate_X, validate_Y, validate_masker)
        return DataLoader(dataset=validate,batch_size=self.batch_size,shuffle=True,)
    
    def test_loader(self,):
        test_X, test_Y, test_masker = self.buildup_data(self.test_index)
        test = AirQualityDataset(test_X, test_Y, test_masker)
        return DataLoader(dataset=test,batch_size=self.batch_size,shuffle=False,)

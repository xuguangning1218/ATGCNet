import os
import torch
import torch.optim as optim
import logging
import numpy as np
from datetime import datetime
from model.ATGCNet import ATGCNet
from model.loss import MultiDistanceLoss
from utility.metrics import MAE, MSE
from dataprovider.dataprovider import AirQualityProvider

class Network:
    
    def __init__(self, config):
        
        # hyper parameters
        self.batch_size = int(config["hyper"]["batch_size"])
        self.learning_rate = float(config["hyper"]["learning_rate"])
        self._lambda1 = float(config["hyper"]["_lambda1"])
        self._lambda2 = float(config["hyper"]["_lambda2"])
        self.margin = float(config["hyper"]["margin"])
        self.mlp_output_dim = int(config["hyper"]["mlp_output_dim"])
        self.hidden_features = int(config["hyper"]["hidden_features"])
        self.out_features = int(config["hyper"]["out_features"])
        self.num_relation = int(config["hyper"]["num_relation"])
        self.num_node = int(config["hyper"]["num_node"])
        self.embeded_dim = int(config["hyper"]["embeded_dim"])
        self.num_layers = int(config["hyper"]["num_layers"])
        
        # model parameters
        self.input_len = int(config["model"]["input_len"])
        self.output_len = int(config["model"]["output_len"])
        self.in_features = int(config["model"]["in_features"])
        self.logger = None
        self.mode = str(config["model"]["mode"])
        
        # only for test
        self.model_save_path = str(config["model"]["model_save_path"])
        self.value = str(config["model"]["value"])
        
        if self.mode == 'train_test':
            self.model_save_folder = './save-{}/'.format(datetime.now().strftime( '%Y%m%d_%H%M%S_%f'))
            if os.path.exists(self.model_save_folder) == False:
                os.makedirs(self.model_save_folder)
                # replace the self.model_save_path for train
                self.model_save_path = self.model_save_folder + 'best_model_{}.pth'.format(self.value)
                self.logger = self.setup_logger(self.model_save_folder)
                params = {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "_lambda1": self._lambda1,
                    "_lambda2": self._lambda2,
                    "margin": self.margin,
                    "mlp_output_dim": self.mlp_output_dim,
                    "hidden_features": self.hidden_features,
                    "out_features": self.out_features,
                    "num_relation": self.num_relation,
                    "num_node": self.num_node,
                    "embeded_dim": self.embeded_dim,
                    "num_layers": self.num_layers,
                }
                for key, item in params.items():
                    self.logger.info("{}:{}".format(key, item))
             
        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # network
        self.model = torch.nn.DataParallel(ATGCNet(in_features=self.in_features, 
                                                    mlp_output_dim=self.mlp_output_dim, 
                                                    hidden_features=self.hidden_features, 
                                                    out_features=self.out_features, 
                                                    num_relation=self.num_relation, 
                                                    num_node=self.num_node, 
                                                    embeded_dim=self.embeded_dim, 
                                                    num_layers=self.num_layers))
        self.model.to(self.device)
        
        
        # loss
        self.criterion1 = torch.nn.L1Loss()
        self.criterion2 = MultiDistanceLoss(margin=self.margin, relation=self.num_relation)
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # data loader
        self.dataprovider = AirQualityProvider(input_len=self.input_len, batch_size=self.batch_size, value=self.value)
        self.train_loader = self.dataprovider.train_loader()
        self.validate_loader = self.dataprovider.validate_loader()
        self.test_loader = self.dataprovider.test_loader()
        
    def train(self, ):
        total_train_loss = 0
        train_cnt = 0
        self.model.train()
        for _data, _target, _ in self.train_loader:
            data = _data.to(self.device)
            target = _target.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            pred, out = self.model(data, )
            mae_loss = self.criterion1(pred, target)
            distance_loss = self.criterion2(out)
            loss = self._lambda1 * mae_loss + self._lambda2 * distance_loss
            total_train_loss += loss.item()
            train_cnt += 1
            loss.backward()
            self.optimizer.step()
        return total_train_loss, train_cnt
    
    def validate(self, ):
        total_val_loss = 0
        val_cnt = 0
        for _data, _target, _ in self.validate_loader:
            data = _data.to(self.device)
            target = _target.to(self.device)
            
            pred, out = self.model(data, )
            mae_loss = self.criterion1(pred, target)
            distance_loss = self.criterion2(out)
            loss = self._lambda1 * mae_loss + self._lambda2 * distance_loss
            
            val_cnt += 1
            total_val_loss += loss.item()
        return total_val_loss, val_cnt
        
    def test(self,):
        
        if self.logger != None:
            print = self.logger.info
        else:
            import builtins
            print = builtins.print
        
        preds = []
        targets = []
        maskers = []

        print('loading best model')
        _, model_dict, _,  _ = self.load()
        self.model.load_state_dict(model_dict)

        print('Best Result:')
        self.model.eval()
        with torch.no_grad():
            for _data, _target, _masker in self.test_loader:
                data = _data.to(self.device)
                pred, _ = self.model(data, )
                target = _target.cpu().detach().numpy()
                masker = _masker.cpu().detach().numpy()
                target = np.squeeze(target, axis=-1)
                pred = np.squeeze(pred, axis=-1)
                pred = pred.cpu().detach().numpy()

                preds.append(pred)
                targets.append(target)
                maskers.append(masker)
        
        preds = np.concatenate(preds, axis=0) * (self.dataprovider._max - self.dataprovider._min) + self.dataprovider._min
        targets = np.concatenate(targets, axis=0) * (self.dataprovider._max - self.dataprovider._min) + self.dataprovider._min
        maskers = np.concatenate(maskers, axis=0) 
           
        for t in range(self.output_len):
            print('predictive time:{}'.format(t))
            mae = MAE(preds[:, t], targets[:, t], maskers[:, t])
            mse = MSE(preds[:, t], targets[:, t], maskers[:, t])
            rmse = np.sqrt(mse)
            print("MAE:{}, MSE:{}, RMSE:{}".format(np.mean(mae), np.mean(mse), np.mean(rmse)))
       
        num_of_test = preds.shape[0]
        reshape_preds = preds.reshape(num_of_test, -1)
        reshape_targets = targets.reshape(num_of_test, -1)
        reshape_maskers = maskers.reshape(num_of_test, -1)
        
        mean_mae = MAE(reshape_preds, reshape_targets, reshape_maskers)
        mean_mse = MSE(reshape_preds, reshape_targets, reshape_maskers)
        mean_rmse = np.sqrt(mean_mse)
        
        print("Mean MAE:{}".format(mean_mae))
        print('Mean MSE:{}'.format(mean_mse))
        print('Mean RMSE:{}'.format(mean_rmse))
    
    def save(self, epoch, validate_loss):
        checkpoint = {
             'epoch': epoch+1, # next epoch
             'model': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(),
             'min_val_loss': validate_loss
        }
        torch.save(checkpoint, self.model_save_path)
        self.logger.info('Saving Model in epoch {}'.format(epoch+1))
    
    def load(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        return checkpoint['epoch'], checkpoint['model'], checkpoint['optimizer'], checkpoint['min_val_loss']
    
    def setup_logger(self, model_save_folder):
        
        level =logging.INFO

        log_name = 'model.log'

        fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
        fileHandler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        fileHandler.setFormatter(formatter)

        logger = logging.getLogger(model_save_folder + log_name)
        logger.setLevel(level)

        logger.addHandler(fileHandler)
    
        return logger
    
    def param_counter(self,):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print('Number of params: %.2fK' % (num_params / 1e3))
        
    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__()
        
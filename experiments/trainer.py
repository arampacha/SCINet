import os
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from metrics.Finantial_metics import MSE, MAE
from experiments.exp_basic import Exp_Basic
from data_process.financial_dataloader import DataLoaderH
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from utils.math_utils import smooth_l1_loss
from models.SCINet import SCINet

from dataclasses import dataclass, asdict


def wandb_available():
    from importlib.util import find_spec
    wandb_spec = find_spec("wandb")
    return wandb_spec is not None


@dataclass
class TrainingArgs:

    verbose:bool=False
    load_best_model:bool=True
    # data
    dataset_name:str='exchange_rate'
    data:str='./exchange_rate.txt'    
    normalize:int=2
    #device
    device:str=None
    use_gpu:bool=True
    use_multi_gpu:bool=False
    gpu:int=0
    # experiment parameters
    window_size:int=168 
    horizon:int=3
    concat_len:int=165
    single_step:int=0 
    single_step_output_One:int=0 
    lastWeight:float=1.0
    # training
    train:bool=True
    resume:bool=False
    evaluate:bool=False
    log_interval:int=100
    metavar='N' 
    save:str='model/model.pt'
    optim:str='adam'
    L1Loss:bool=True
    num_nodes:int=1
    batch_size:int=8
    lr:float=5e-3
    weight_decay:float=0.00001
    epochs:int=100
    lradj:int=1
    save_path:str='sp500/'
    model_name:str='SCINet'
    # model
    input_dim:int=5
    hidden_size:int=1
    INN:int=1
    kernel:int=5
    dilation:int=1
    positionalEcoding:bool=False
    dropout:float=0.5
    groups:int=1
    levels:int=3
    num_decoder_layer:int=1
    stacks:int=1
    long_term_forecast:bool=False
    RIN:bool=False


    def __post_init__(self):
        if not self.long_term_forecast:
            self.concat_len = self.window_size - self.horizon
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device=torch.device(self.device)


class Trainer(Exp_Basic):
    """
    Based on original SCINet authors code https://github.com/cure-lab/SCINet
    """

    def __init__(self, args):
        super().__init__(args)
        if self.args.L1Loss:
            self.criterion = smooth_l1_loss
        else:
            self.criterion = nn.MSELoss(size_average=False).to(args.device)
        self.evaluateL2 = nn.MSELoss(size_average=False).to(args.device)
        self.evaluateL1 = nn.L1Loss(size_average=False).to(args.device)
        self.writer = SummaryWriter('.exp/run_financial/{}'.format(args.model_name))
    
    def _build_model(self):
            
        model = SCINet(
            output_len=self.args.horizon,
            input_len=self.args.window_size,
            input_dim=self.args.input_dim,
            hid_size=self.args.hidden_size,
            num_stacks=self.args.stacks,
            num_levels=self.args.levels,
            num_decoder_layer=self.args.num_decoder_layer,
            concat_len=self.args.concat_len,
            groups=self.args.groups,
            kernel=self.args.kernel,
            dropout=self.args.dropout,
            single_step_output_One=self.args.single_step_output_One,
            positionalE=self.args.positionalEcoding,
            modified=True,
            RIN=self.args.RIN
        )
        if self.args.verbose:
            print(model)
        return model
    
    def save_model(self, save_path=None, epoch=0, lr=0.):
        if save_path is None:
            save_path = os.path.join(self.args.save_path, self.args.model_name)
        save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
    
    def _get_data(self):

        if self.args.long_term_forecast:
            return DataLoaderH(self.args.data, 0.7, 0.1, self.args.horizon, self.args.window_size, 4, device=self.args.device)
        else:
            return DataLoaderH(self.args.data, 0.6, 0.2, self.args.horizon, self.args.window_size, self.args.normalize, device=self.args.device)

    def _select_optimizer(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    
    def _init_wandb(self):
        if wandb_available():
            import wandb
            self._wandb = wandb
            self.run = wandb.init(
                entity=os.environ.get("WANDB_ENTITY"),
                project=os.environ.get("WANDB_PROJECT"),
                name=f"{self.args.model_name}-{self.args.dataset_name}-{self.args.window_size}-{self.args.horizon}",
                config=asdict(self.args)
            )

    def train(self):

        best_val=10000000
        
        optim=self._select_optimizer()

        data=self._get_data()
        X=data.train[0]
        Y=data.train[1]
        save_path = os.path.join(self.args.save_path, self.args.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        self._init_wandb()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
        else:
            epoch_start = 0
            
        bs = self.args.batch_size
        steps_per_epoch = math.ceil(len(X)/bs)
        for epoch in trange(epoch_start, self.args.epochs):
            epoch_start_time = time.time()
            iter = 0
            self.model.train()
            total_loss = 0
            n_samples = 0
            final_loss = 0
            min_loss = 0
            lr = adjust_learning_rate(optim, epoch, self.args)
            
            pbar = tqdm(
                data.get_batches(X, Y, bs, True), total=steps_per_epoch, leave=False
            )
            for tx, ty in pbar:
                self.model.zero_grad()
                if self.args.stacks == 1:
                    forecast = self.model(tx)
                elif self.args.stacks == 2: 
                    forecast, res = self.model(tx)
                scale = data.scale.expand(forecast.size(0), self.args.horizon, data.m)
                bias = data.bias.expand(forecast.size(0), self.args.horizon, data.m)
                weight = torch.tensor(self.args.lastWeight).to(self.args.device) #used with multi-step

                if self.args.single_step: #single step
                    ty_last = ty[:, -1, :]
                    scale_last = data.scale.expand(forecast.size(0), data.m)
                    bias_last = data.bias.expand(forecast.size(0), data.m)
                    if self.args.normalize == 3:
                        loss_f = self.criterion(forecast[:, -1], ty_last)
                        if self.args.stacks == 2:
                            loss_m = self.criterion(res, ty)/res.shape[1] #average results

                    else:
                        loss_f = self.criterion(forecast[:, -1] * scale_last + bias_last, ty_last * scale_last + bias_last)
                        if self.args.stacks == 2:
                            loss_m = self.criterion(res * scale + bias, ty * scale + bias)/res.shape[1] #average results

                else:
                    if self.args.normalize == 3:
                        if self.args.lastWeight == 1.0:
                            loss_f = self.criterion(forecast, ty)
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res, ty)
                        else:
                            loss_f = self.criterion(forecast[:, :-1, :], ty[:, :-1, :] ) \
                                    + weight * self.criterion(forecast[:, -1:, :], ty[:, -1:, :] )
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res[:, :-1, :] , ty[:, :-1, :] ) \
                                        + weight * self.criterion(res[:, -1:, :], ty[:, -1:, :] )
                    else:
                        if self.args.lastWeight == 1.0:
                            loss_f = self.criterion(forecast * scale + bias, ty * scale + bias)
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res * scale + bias, ty * scale + bias)
                        else:
                            loss_f = self.criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                            ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                                + weight * self.criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                        ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                                ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                                    + weight * self.criterion(res[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                            ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                loss = loss_f
                if self.args.stacks == 2:
                    loss += loss_m

                loss.backward()
                total_loss += loss.item()

                final_loss  += loss_f.item()
                if self.args.stacks == 2:
                    min_loss  += loss_m.item()
                n_samples += (forecast.size(0) * data.m)
                grad_norm = optim.step()

                pbar.set_description(desc='loss: {:.7f}'.format(loss.item()/(forecast.size(0) * data.m)))
                if iter%100==0:
                    self._wandb.log(
                        {"train_loss_total":loss.item()/(forecast.size(0) * data.m),
                        "train_loss_final":loss_f.item()/(forecast.size(0) * data.m),
                         "learning_rate":lr},
                        step = epoch*steps_per_epoch + iter
                        )
                iter += 1
            if self.args.stacks == 1:
                val_loss, val_rae, val_corr, val_mse, val_mae, val_trend_acc = self.validate(data, data.valid[0],data.valid[1])
            elif self.args.stacks == 2:
                val_loss, val_rae, val_corr, val_rse_mid, val_rae_mid, val_correlation_mid, val_mse, val_mae, val_trend_acc =self.validate(data, data.valid[0],data.valid[1])


            self.writer.add_scalar('Train_loss_total', total_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Train_loss_final', final_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Validation_final_rse', val_loss, global_step=epoch)
            self.writer.add_scalar('Validation_final_rae', val_rae, global_step=epoch)
            self.writer.add_scalar('Validation_final_corr', val_corr, global_step=epoch)
            self.writer.add_scalar('Validation_trend_acc', val_trend_acc, global_step=epoch)
            if self.args.stacks == 2:
                self.writer.add_scalar('Train_loss_Mid', min_loss / n_samples, global_step=epoch)
                self.writer.add_scalar('Validation_mid_rse', val_rse_mid, global_step=epoch)
                self.writer.add_scalar('Validation_mid_rae', val_rae_mid, global_step=epoch)
                self.writer.add_scalar('Validation_mid_corr', val_correlation_mid, global_step=epoch)
            
            metrics = [total_loss / n_samples, final_loss/n_samples, val_loss, val_rae, val_corr, val_mse, val_mae, val_trend_acc, lr, epoch+1]
            metric_names=["train_loss_total", "train_loss_final", "val_loss", "val_rae", "val_corr", "val_mse", "val_mae", "val_trend_acc", "learning_rate", "epoch"]
            metrics_dict={k:v for k,v in zip(metric_names, metrics)}
            self._wandb.log(metrics_dict, step=epoch*steps_per_epoch+iter)
            if val_mse < best_val and self.args.long_term_forecast:
                save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
                # print('--------------| Best Val loss |--------------')
                best_val = val_mse
            elif val_loss < best_val and not self.args.long_term_forecast:
                save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
                # print('--------------| Best Val loss |--------------')
                best_val = val_loss

        if self.args.load_best_model:
            print("Loading best model checkpoint...", end=" ")
            self.model = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)[0]
            print("Done")

        return total_loss / n_samples

    def validate(self, data, X, Y, evaluate=False):
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0

        total_loss_mid = 0
        total_loss_l1_mid = 0
        n_samples = 0
        predict = None
        res_mid = None
        test = None

        forecast_set = []
        Mid_set = []
        target_set = []
        last_x = []

        if evaluate:
            save_path = os.path.join(self.args.save_path, self.args.model_name)
            self.model = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)[0]

        for X, Y in data.get_batches(X, Y, self.args.batch_size, False):
            last_x.append(X[:, -1, :].cpu())
            with torch.no_grad():
                if self.args.stacks == 1:
                    forecast = self.model(X)
                elif self.args.stacks == 2:
                    forecast, res = self.model(X) #torch.Size([32, 3, 137])
            # only predict the last step
            true = Y[:, -1, :].squeeze()
            output = forecast[:,-1,:].squeeze()

            forecast_set.append(forecast)
            target_set.append(Y)
            if self.args.stacks == 2:
                Mid_set.append(res)

            if len(forecast.shape)==1:
                forecast = forecast.unsqueeze(dim=0)
                if self.args.stacks == 2:
                    res = res.unsqueeze(dim=0)
            if predict is None:
                predict = forecast[:,-1,:].squeeze()
                test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])
                if self.args.stacks == 2:
                    res_mid = res[:,-1,:].squeeze()

            else:
                predict = torch.cat((predict, forecast[:,-1,:].squeeze()))
                test = torch.cat((test, Y[:, -1, :].squeeze()))
                if self.args.stacks == 2:
                    res_mid = torch.cat((res_mid, res[:,-1,:].squeeze()))
            
            scale = data.scale.expand(output.size(0),data.m)
            bias = data.bias.expand(output.size(0), data.m)
            if self.args.stacks == 2:
                output_res = res[:,-1,:].squeeze()

            total_loss += self.evaluateL2(output * scale + bias, true * scale+ bias).item()
            total_loss_l1 += self.evaluateL1(output * scale+ bias, true * scale+ bias).item()
            if self.args.stacks == 2:
                total_loss_mid += self.evaluateL2(output_res * scale+ bias, true * scale+ bias).item()
                total_loss_l1_mid += self.evaluateL1(output_res * scale+ bias, true * scale+ bias).item()

            n_samples += (output.size(0) * data.m)
        
        last_x = torch.cat(last_x, axis=0).numpy()
        forecast_Norm = torch.cat(forecast_set, axis=0)
        target_Norm = torch.cat(target_set, axis=0)
        mse = MSE(forecast_Norm.cpu().numpy(), target_Norm.cpu().numpy())
        mae = MAE(forecast_Norm.cpu().numpy(), target_Norm.cpu().numpy())

        if self.args.stacks == 2:
            Mid_Norm = torch.cat(Mid_set, axis=0)

        rse_final_each = []
        rae_final_each = []
        corr_final_each = []
        Scale = data.scale.expand(forecast_Norm.size(0),data.m)
        bias = data.bias.expand(forecast_Norm.size(0),data.m)
        if not self.args.single_step: #single step
            for i in range(forecast_Norm.shape[1]): #get results of each step
                lossL2_F = self.evaluateL2(forecast_Norm[:,i,:] * Scale + bias, target_Norm[:,i,:] * Scale+ bias).item()
                lossL1_F = self.evaluateL1(forecast_Norm[:,i,:] * Scale+ bias, target_Norm[:,i,:] * Scale+ bias).item()
                if self.args.stacks == 2:
                    lossL2_M = self.evaluateL2(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
                    lossL1_M = self.evaluateL1(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
                rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0]/ data.m) / data.rse
                rae_F = (lossL1_F / forecast_Norm.shape[0]/ data.m) / data.rae
                rse_final_each.append(rse_F.item())
                rae_final_each.append(rae_F.item())

                pred = forecast_Norm[:,i,:].data.cpu().numpy()
                y_true = target_Norm[:,i,:].data.cpu().numpy()

                sig_p = pred.std(axis=0)
                sig_g = y_true.std(axis=0)
                m_p = pred.mean(axis=0)
                m_g = y_true.mean(axis=0)
                ind = (sig_p * sig_g != 0)
                corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
                corr = (corr[ind]).mean()
                corr_final_each.append(corr)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae
        if self.args.stacks == 2:
            rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
            rae_mid = (total_loss_l1_mid / n_samples) / data.rae

        # only calculate the last step for financial datasets.
        predict = forecast_Norm.cpu().numpy()[:,-1,:]
        Ytest = target_Norm.cpu().numpy()[:,-1,:]
        
        #trand accuracy
        trend_true = Ytest > last_x
        trend_pred = predict > last_x
        trend_acc = (trend_true == trend_pred).mean()

        sigma_p = predict.std(axis=0)
        sigma_g = Ytest.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_p * sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        if self.args.stacks == 2:
            mid_pred = Mid_Norm.cpu().numpy()[:,-1,:]
            sigma_mid = mid_pred.std(axis=0)
            mean_mid = mid_pred.mean(axis=0)
            index_mid = (sigma_mid * sigma_g != 0)
            correlation_mid = ((mid_pred - mean_mid) * (Ytest - mean_g)).mean(axis=0) / (sigma_mid * sigma_g)
            correlation_mid = (correlation_mid[index_mid]).mean()

        print(
            '|valid_final mse {:5.4f} |valid_final mae {:5.4f} |valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}| trend_acc {:5.4f}'.format(mse,mae,
                rse, rae, correlation, trend_acc), flush=True)
        if self.args.stacks == 2:
            print(
            '|valid_final mse {:5.4f} |valid_final mae {:5.4f} |valid_mid rse {:5.4f} | valid_mid rae {:5.4f} | valid_mid corr  {:5.4f}'.format(mse,mae,
                rse_mid, rae_mid, correlation_mid), flush=True)

        if self.args.stacks == 1:
            return rse, rae, correlation, mse, mae, trend_acc
        if self.args.stacks == 2:
            return rse, rae, correlation, rse_mid, rae_mid, correlation_mid, mse, mae, trend_acc

    
    def predict(self, data, X, Y, scale=True):
        self.model.eval()
        predictions = []
        total_steps = math.ceil(len(X)/self.args.batch_size)
        for xb, yb in tqdm(data.get_batches(X, Y, self.args.batch_size, False), total=total_steps):
            with torch.no_grad():
                if self.args.stacks == 1:
                    forecast = self.model(xb)
                elif self.args.stacks == 2:
                    forecast, res = self.model(xb)
            predictions.append(forecast.cpu())
        
        predictions = torch.cat(predictions, axis=0)

        if scale:
            predictions = data.scale.cpu() * predictions + data.bias.cpu()

        return predictions
    
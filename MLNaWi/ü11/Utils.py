import copy
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage.filters import uniform_filter1d
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def update(a, b):
    for i, k in enumerate(a.keys()):
        a[k].append(b[i])


def plot_reconstr(model: nn.Module, data_loader: DataLoader, n_samples: int = 5):
    print(f"Let's plot some reconstructed data: ")
    for batch in data_loader:
        data = batch['data']
        output = model(data)
        reconstr = output[0]
        for i, d in enumerate(zip(data.detach().numpy(), reconstr.detach().numpy())):
            if i >= n_samples:
                break
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
            ax1.plot(np.arange(len(d[0].squeeze())), d[0].squeeze())
            ax2.plot(np.arange(len(d[1].squeeze())), d[1].squeeze())
            ax1.set_title('ground truth')
            ax2.set_title('reconstructed')
            fig.show()
        #			plt.close()
        break


def get_data(batch_size: int = 128, feat_length: int = 350, N: int = 5,
             composed: bool = False):
    def transform(df):
        X = df.iloc[:, :feat_length].values
        X /= scaling_const
        X = uniform_filter1d(X, size=N)
        return X

    if composed:
        df = pd.read_hdf('ECG.h5')
        df = df[df['label'].isin(['Normal', 'PVC'])]

        X = df.iloc[:, :feat_length].values
        max_positive_amplitude = X.max()
        max_negative_amplitude = X.min()
        scaling_const = max(max_positive_amplitude, abs(max_negative_amplitude))
        X /= scaling_const
        y = df['label'].astype('category').cat.codes.values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        #		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
        #		test_size=0.2)

        train = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size,
                           shuffle=True)
        val = DataLoader(MyDataset(X_val, y_val), batch_size=batch_size)
        #		test = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size)
        return train, val, val

    df = pd.read_hdf('ECG.h5')
    normal = df['label'] == 'Normal'
    abnormal = df['label'] == 'PVC'
    #	abnormal = ~normal
    # scaling
    max_positive_amplitude = df[normal].iloc[:, :feat_length].max().max()
    max_negative_amplitude = df[normal].iloc[:, :feat_length].min().min()
    scaling_const = max(max_positive_amplitude, abs(max_negative_amplitude))
    # extract numpy arrays and split
    X_normal = transform(df[normal])
    X_train, X_val = train_test_split(X_normal, test_size=0.2)
    y_train, y_val = np.zeros(len(X_train)), np.zeros(len(X_val))
    X_abnormal = transform(df[abnormal])
    _, X_test = train_test_split(X_abnormal, test_size=0.1)
    y_test = np.ones(len(X_test))

    X_test = np.concatenate((X_test, X_val[:650]))
    y_test = np.concatenate((y_test, y_val[:650]))
    X_val = X_val[650:]
    y_val = y_val[650:]

    train = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val = DataLoader(MyDataset(X_val, y_val), batch_size=batch_size)
    test = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size)
    return train, val, test


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.array, y: np.array):
        self.data = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(y).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return {'data': x, 'labels': y}

    def __len__(self):
        return len(self.labels)


class AbstractTrainer(ABC):
    def __init__(self, model: nn.Module, train_loader: DataLoader = None,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None, optimizer=None,
                 epochs: int = 100,
                 kl_scaling: float = 0.5, reconstr_loss: str = 'L1',
                 early_stop: Optional[int] = None):
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)
        if reconstr_loss == 'L2':
            self.reconstr_loss = nn.MSELoss(reduction='none')
        elif reconstr_loss == 'L1':
            self.reconstr_loss = nn.L1Loss(reduction='none')
        self.epochs = epochs
        self.kl_scaling = kl_scaling
        self.early_stop = early_stop  # or int indicating index of measure key

        # if you have a gpu you can uncomment next line and comment the following one,
        # everything should work fine but I haven't tested it -> be careful
        #		self.device = torch.device("cuda" if torch.cuda.is_available() else
        #		"cpu")
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def _save(self):
        self.best_model = self.model.state_dict()
        print('Model saved')

    def _plot_training(self, logger):
        print(f'Summary of training:')
        colors = {0: 'blue', 1: 'orange'}
        datasets = list(logger.keys())
        measures = list(logger[datasets[0]].keys())
        fig, axs = plt.subplots(nrows=len(measures), ncols=1, sharex=True, figsize=(5, 7))
        for idx, measure in enumerate(measures):
            for col, dataset in enumerate(datasets):
                axs[idx].plot(np.arange(self.epochs), logger[dataset][measure],
                              c=colors[col], label=dataset)
                axs[idx].set_title(measure)
        axs[-1].legend()
        plt.tight_layout()
        fig.show()

    #		plt.close()

    def _out_string(self, dataset: str, vals: list):
        out_string = str()
        for i, m in enumerate(self.measures.keys()):
            out_string += m + ' : ' + str(round(vals[i], 6)) + ' '
        return dataset + '\t' + out_string

    def _step(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        res = self._loss(data, label, *output)
        loss = res[0]
        loss.backward()
        self.optimizer.step()
        reconstructed = output[0]
        return [elem.item() for elem in res], reconstructed

    def train(self, plot: bool = True):
        best_validation_error = np.inf
        logger = {'train': copy.deepcopy(self.measures),
                  'val': copy.deepcopy(self.measures)}
        for epoch in range(self.epochs):
            self.ep = epoch
            print("\n===> epoch: %d" % epoch)
            self.model.train()
            train_log = np.zeros(len(self.measures))
            for batch in self.train_loader:
                data = batch['data'].to(self.device)
                label = batch['labels'].to(self.device).squeeze()
                res, out = self._step(data, label)
                train_log += np.array(res)
            train_log /= len(self.train_loader)
            print(self._out_string('Train', train_log))
            val_log = self._validate()
            ## early stopping on reconstruction error
            if self.early_stop:
                if val_log[self.early_stop] <= best_validation_error:
                    best_validation_error = val_log[self.early_stop]
                    self._save()
            update(logger['train'], train_log)
            update(logger['val'], val_log)
            #scheduler
            if self.optimizer.param_groups[0]['lr'] > 1e-7:
                self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < 1e-7:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-6
        if self.early_stop is not None:
            measure = list(self.measures.keys())[self.early_stop]
            print("\n Best %s on validation: %.6f" % (measure, best_validation_error))
            self.model.load_state_dict(self.best_model)
        if plot:
            self._plot_training(logger)
        return self.model

    def _validate(self):
        self.model.eval()
        val_log = np.zeros(len(self.measures))
        with torch.no_grad():
            for batch in self.val_loader:
                data = batch['data'].to(self.device)
                label = batch['labels'].to(self.device).squeeze()
                output = self.model(data)
                res = self._loss(data, label, *output)
                val_log += np.array([elem.item() for elem in res])
            val_log /= len(self.val_loader)
        print(self._out_string('Val', val_log))
        return val_log

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def test(self):
        pass

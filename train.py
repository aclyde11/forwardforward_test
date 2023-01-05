import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models import SimpleDenseNet
from models import ForwardForwardNet
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm


def train_ff(trainData, testData, input_dim=2048, num_iters=100, batch_size=64, lr=0.001, dropout=0.0,
                hidden_dims=None, device_str='cpu'):
    device = torch.device(device_str)
    if hidden_dims is None:
        hidden_dims = [256, 128]

    ff_model = ForwardForwardNet(input_dim=input_dim, hidden_dims=hidden_dims, device=device, dropout=dropout).to(device)

    ff_net_performance = []

    data_loader_kwargs = {'batch_size': batch_size,
                          'shuffle': True,
                          'num_workers': (0 if trainData.pre_compute else 8)}
    if device == 'cuda':
        data_loader_kwargs['pin_memory'] = True
        data_loader_kwargs['pin_memory_device'] = device_str
    train_dataloader = DataLoader(trainData, **data_loader_kwargs)
    test_dataloader = DataLoader(testData, **data_loader_kwargs)

    for iter_ in range(num_iters):
        ff_model.train()
        for X_batch_pos, X_batch_neg, _, _ in tqdm(train_dataloader, total=len(train_dataloader)):
            X_batch_pos = X_batch_pos.float().to(device)
            X_batch_neg = X_batch_neg.float().to(device)

            ff_model.ff_train(X_batch_pos, X_batch_neg)

        if True:
            ff_model.eval()
            y_test_all = []
            y_pred_all = []
            with torch.no_grad():
                for X_batch, y_batch in tqdm(test_dataloader, total=len(test_dataloader)):
                    X_batch = X_batch.float().to(device)
                    y_batch = y_batch.float().to(device)

                    y_pred = ff_model.predict(X_batch)
                    y_pred = y_pred.detach().cpu().numpy()
                    y_test_all.append(y_batch.cpu().numpy().flatten())
                    y_pred_all.append(y_pred.flatten())
            ff_net_performance.append(roc_auc_score(np.concatenate(y_test_all).flatten(), np.concatenate(y_pred_all).flatten()))
            print(f'Iter: {iter_}, Test ROC-AUC: {ff_net_performance[-1]}')
    return ff_model, ff_net_performance,  np.concatenate(y_pred_all).flatten(), np.concatenate(y_test_all).flatten()


def train_dense(trainData, testData, input_dim=2048, num_iters=100, batch_size=64, lr=0.0001, dropout=0.05,
                hidden_dims=None, device_str='cpu'):
    device = torch.device(device_str)
    if hidden_dims is None:
        hidden_dims = [256, 128]

    dense_model = SimpleDenseNet(input_dim=input_dim, output_dim=2,
                                 hidden_dims=hidden_dims, dropout=dropout).to(device)

    optimizer = optim.Adam(dense_model.parameters(), lr=lr)
    simple_net_performance = []
    loss_ce = nn.CrossEntropyLoss()

    data_loader_kwargs = {'batch_size': batch_size,
                          'shuffle': True,
                          'num_workers': (0 if trainData.pre_compute else 8)}
    if device == 'cuda':
        data_loader_kwargs['pin_memory'] = True
        data_loader_kwargs['pin_memory_device'] = device_str
    train_dataloader = DataLoader(trainData, **data_loader_kwargs)
    test_dataloader = DataLoader(testData, **data_loader_kwargs)

    for iter_ in range(num_iters):
        dense_model.train()
        for X_batch_pos, X_batch_neg, y_batch_pos, y_batch_neg in tqdm(train_dataloader, desc=f"Train {iter_}", total=len(train_dataloader)):
            optimizer.zero_grad()
            X_out = torch.cat([X_batch_pos, X_batch_neg], dim=0).float().to(device)
            y_out = torch.cat([y_batch_pos, y_batch_neg], dim=0).float().to(device)

            out = dense_model(X_out)
            loss = loss_ce(out.squeeze(), y_out.long())
            loss.backward()
            optimizer.step()
        if True:
            dense_model.eval()
            tmp = []
            y_pred_all = []
            y_test_all = []
            with torch.no_grad():
                for X_batch, y_test in tqdm(test_dataloader, desc=f"Eval {iter_}", total=len(test_dataloader)):
                    X_batch = X_batch.float().to(device)
                    y_test = y_test.float().to(device)
                    out = dense_model(X_batch).float()
                    y_pred = (torch.max(out, 1)[1]).cpu()
                    y_pred = np.squeeze(np.asarray(y_pred.numpy()))
                    y_test = y_test.cpu().numpy()
                    y_pred_all.append(y_pred)
                    y_test_all.append(y_test)
            simple_net_performance.append(roc_auc_score(np.concatenate(y_test_all).flatten(), np.concatenate(y_pred_all).flatten()))
            print(f'Iter: {iter_}, Test ROC-AUC: {simple_net_performance[-1]}')
    return dense_model, simple_net_performance,  np.concatenate(y_pred_all).flatten(), np.concatenate(y_test_all).flatten()

import torch
import random
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim

import sys
sys.path.append('../..')

from model import Net
from utils.config import process_config, get_args
import datetime
from utils.util import warm_up_lr, flag


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


class In:
    def readline(self):
        return "y\n"

    def close(self):
        pass


def train(model, device, loader, optimizer, task_type, config):
    model.train()
    loss_all = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            criterion =  cls_criterion if "classification" in task_type else reg_criterion
            if config.flag == 'true':
                forward = lambda perturb : model(batch, perturb).to(torch.float32)[is_labeled]
                model_forward = (model, forward)
                y = batch.y.to(torch.float32)[is_labeled]
                perturb_shape = (batch.x.shape[0], model.config.hidden)
                loss, _ = flag(model_forward, perturb_shape, y, optimizer, device, criterion)
                loss_all += loss.item()
            else:
                pred = model(batch)
                optimizer.zero_grad()
                loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward()
                loss_all += loss.item()
                optimizer.step()

    return loss_all / len(loader)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting

    #sys.stdin = In()

    dataset = PygGraphPropPredDataset(name=config.dataset_name)

    if config.feature == 'full':
        pass
    elif config.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(config.dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers)

    model = Net(config.architecture, num_tasks=dataset.num_tasks).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    
    #optimizer = optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate, weight_decay=config.hyperparams.weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.hyperparams.epochs - config.hyperparams.warmup_epochs, eta_min=config.hyperparams.learning_rate/100)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparams.step_size,
                                                gamma=config.hyperparams.decay_rate)
    
    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []
    
    for epoch in range(1, config.hyperparams.epochs + 1):
        if epoch <= config.hyperparams.warmup_epochs:
            warm_up_lr(epoch, config.hyperparams.warmup_epochs, config.hyperparams.learning_rate, optimizer)
        print (datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print("Epoch {} training...".format(epoch))
        print ("lr: ", optimizer.param_groups[0]['lr'])
        train_loss = train(model, device, train_loader, optimizer, dataset.task_type, config.architecture)
        if epoch > config.hyperparams.warmup_epochs:
            scheduler.step()

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print('Train:', train_perf[dataset.eval_metric],
              'Validation:', valid_perf[dataset.eval_metric],
              'Test:', test_perf[dataset.eval_metric],
              'Train loss:', train_loss)

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        trainL_curve.append(train_loss)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'
          .format(test_curve[-1], valid_curve[-1],
                  epoch-1, best_train, min(trainL_curve)))

if __name__ == "__main__":
    main()

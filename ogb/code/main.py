import random
import torch
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import sys
sys.path.append('../..')

### importing utils
from proc import ASTNodeEncoder, get_vocab_mapping
### for data transform
from proc import augment_edge, encode_y_to_arr, decode_arr_to_seq

from model import Net
from utils.config import process_config, get_args
import datetime
from utils.util import warm_up_lr, flag


class In:
    def readline(self):
        return "y\n"

    def close(self):
        pass


def train(model, device, loader, optimizer, multicls_criterion):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss = loss / len(pred_list)

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    #if config.get('seed') is not None:
    #    random.seed(config.seed)
    #    torch.manual_seed(config.seed)
    #    np.random.seed(config.seed)
    #    if torch.cuda.is_available():
    #        torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting

    #sys.stdin = In()

    dataset = PygGraphPropPredDataset(name=config.dataset_name)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(config.max_seq_len, np.sum(seq_len_list <= config.max_seq_len) / len(seq_len_list)))

    split_idx = dataset.get_idx_split()

    ### building vocabulary for sequence predition. Only use training data.

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], config.num_vocab)

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, config.max_seq_len)])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(config.dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.hyperparams.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=config.num_workers)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(config.architecture.hidden, num_nodetypes=len(nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    model = Net(config.architecture,
                num_vocab=len(vocab2idx),
                max_seq_len=config.max_seq_len,
                node_encoder=node_encoder).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    #optimizer = optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate, weight_decay=config.hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparams.step_size,
                                                gamma=config.hyperparams.decay_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.hyperparams.epochs - config.hyperparams.warmup_epochs)

    multicls_criterion = torch.nn.CrossEntropyLoss()

    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []

    #writer = SummaryWriter(config.directory)

    #ts_fk_algo_hp = str(config.time_stamp) + '_' \
    #                + str(config.commit_id[0:7]) + '_' \
    #                + str(config.architecture.methods) + '_' \
    #                + str(config.architecture.pooling) + '_' \
    #                + str(config.architecture.JK) + '_' \
    #                + str(config.architecture.layers) + '_' \
    #                + str(config.architecture.hidden) + '_' \
    #                + str(config.architecture.variants.BN) + '_' \
    #                + str(config.architecture.dropout) + '_' \
    #                + str(config.hyperparams.learning_rate) + '_' \
    #                + str(config.hyperparams.step_size) + '_' \
    #                + str(config.hyperparams.decay_rate) + '_' \
    #                + 'B' + str(config.hyperparams.batch_size) + '_' \
    #                + 'S' + str(config.seed if config.get('seed') is not None else "na") + '_' \
    #                + 'W' + str(config.num_workers if config.get('num_workers') is not None else "na")

    for epoch in range(1, config.hyperparams.epochs + 1):
        if epoch <= config.hyperparams.warmup_epochs:
            warm_up_lr(epoch, config.hyperparams.warmup_epochs, config.hyperparams.learning_rate, optimizer)
        print (datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print("Epoch {} training...".format(epoch))
        print ("lr: ", optimizer.param_groups[0]['lr'])
        train_loss = train(model, device, train_loader, optimizer, multicls_criterion)

        if epoch > config.hyperparams.warmup_epochs:
            scheduler.step()

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        valid_perf = eval(model, device, valid_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        test_perf = eval(model, device, test_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

        print('Train:', train_perf[dataset.eval_metric],
              'Validation:', valid_perf[dataset.eval_metric],
              'Test:', test_perf[dataset.eval_metric],
              'Train loss:', train_loss)

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        trainL_curve.append(train_loss)

        #writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/traP': train_perf[dataset.eval_metric]}, epoch)
        #writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/valP': valid_perf[dataset.eval_metric]}, epoch)
        #writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/tstP': test_perf[dataset.eval_metric]}, epoch)
        #writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/traL': train_loss}, epoch)
    #writer.close()

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    print('Finished test: {}, Validation: {}, Train: {}, epoch: {}, best train: {}, best loss: {}'
          .format(test_curve[best_val_epoch], valid_curve[best_val_epoch], train_curve[best_val_epoch],
                  best_val_epoch, best_train, min(trainL_curve)))


if __name__ == "__main__":
    main()

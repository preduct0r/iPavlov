import argparse
import os
import os.path as osp
import random
import tqdm
import time

import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from Task_1_miffka.assignment1.config import config
from Task_1_miffka.assignment1.dataset import SkipGramDataset, Batcher, BatcherNS, collate_fn
from Task_1_miffka.assignment1.network import SkipGram, SkipGramNS


def train_one_epoch(model, loader, optimizer, epoch, device, neg_sampling=False, \
                    log_interval=10, verbose=False, writer=None):
    model.to(device)
    model.train()
    posr_loss, negr_loss, r_loss = 0, 0, 0
    
    n_steps = len(loader)
    for i, batch in tqdm.tqdm(enumerate(loader), desc=f'Epoch {epoch}', total=n_steps):
        pos_loss, neg_loss = model(*batch, device)
        loss = (pos_loss + neg_loss)
        posr_loss += pos_loss.item()
        negr_loss += neg_loss.item()
        r_loss += loss.item()
        if not(i % log_interval):
            if verbose:
                print(f'TOTAL\t{r_loss/(i+1)}\tPOS\t{posr_loss/(i+1)}\tNEG\t{negr_loss/(i+1)}')
            if writer:
                writer.add_scalar('TOTAL', r_loss/(i+1), global_step=i+n_steps*epoch)
                writer.add_scalar('POS', posr_loss/(i+1), global_step=i+n_steps*epoch)
                writer.add_scalar('NEG', negr_loss/(i+1), global_step=i+n_steps*epoch)
                
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save({'state': model.state_dict(),
                'int2token': dataset.int2token,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, 
                osp.join(config.model_dir, args.task_name, f'model_{epoch}.pth'))
    return r_loss/(i+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network via Skip-Gram design')
    # Common settings
    parser.add_argument('--task_name', type=str, default='sample_task',
                        help='Name of the task')
    parser.add_argument('--negative_sampling', action='store_true',
                        help='Train model with negative sampling')

    # Data settings
    parser.add_argument('--text_file', type=str, default=osp.join(config.data_dir, 'text8'),
                        help='Path to text file')
    parser.add_argument('--dict_size', type=int, default=100000,
                        help='Size of the dictionary - how many unique words to fetch')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimal count of the single word to be included in dictionary')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Size of the window for context retrieval')
    parser.add_argument('--neg_sample_n', type=int, default=10,
                        help='Number of the negative samples for each word')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Batch size')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode (for debugging)')
    parser.add_argument('--test_size', type=int, default=20000,
                        help='Total number of tokens in corpus for test mode')

    # Model settings
    parser.add_argument('--dim_size', type=int, default=200,
                        help='Dimension of the resulting word2vec model')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Batch size')
    parser.add_argument('--force_cpu', action='store_false', dest='use_gpu',
                        help='Whether to use cuda device')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to saved model & optimizer state')

    # Common settings
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Log metrics each log_interval\'th batch')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to print metric values to console output')
    parser.add_argument('--random_state', type=int, default=24,
                        help='Random seed')

    parser.set_defaults(use_gpu=True)
    args = parser.parse_args()

    # Set up Tensorboard Writer
    os.makedirs(f'{config.logs_dir}/{args.task_name}', exist_ok=True)
    writer = SummaryWriter(f'{config.logs_dir}/{args.task_name}/{time.strftime("%Y-%m-%d_%H:%M:%S")}')

    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Fixate random seeds
    random.seed(args.random_state)
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize dataset, sampler, and loader
    dataset = SkipGramDataset(args.text_file, dict_size=args.dict_size, min_count=args.min_count,
                                window_size=args.window_size, neg_sample_n=args.neg_sample_n, 
                                test_mode=args.test_mode, test_size=args.test_size)

    sampler = RandomSampler(dataset)
    if not args.negative_sampling:
        batcher = Batcher(dataset, batch_size=args.batch_size, sampler=sampler, 
                          num_workers=args.num_workers, collate_fn=collate_fn, drop_last=drop_last)
    else:
        batcher = BatcherNS(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print('Batcher initialized')

    # Initalize model and optimizer
    if not args.negative_sampling:
        model = SkipGram(dataset.dict_size, args.dim_size)
    else:
        model = SkipGramNS(dataset.dict_size, args.dim_size)
    optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=0)
    print('Model and optimizer initialized')

    # Load model and optimizer state if checkpoint is provided
    if args.checkpoint is not None:
        checkpoint_dict = torch.load(args.checkpoint, map_location='cpu')
        if 'state' in checkpoint_dict:
            model.load_state_dict(checkpoint_dict['state'])
        if 'optimizer' in checkpoint_dict:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        init_epoch = checkpoint_dict.get('epoch', 0) + 1
        print(f'Checkpoint {args.checkpoint} loaded')
    else:
        init_epoch = 0

    # Train loop
    os.makedirs(osp.join(config.model_dir, args.task_name), exist_ok=True)
    best_loss = None
    for epoch in range(init_epoch, args.num_epochs + init_epoch):
        loss = train_one_epoch(model, batcher, optimizer, epoch, device=device, neg_sampling=args.negative_sampling,
                               log_interval=args.log_interval, verbose=args.verbose, writer=writer)
        if best_loss is None or loss < best_loss:
            torch.save({'state': model.state_dict(),
                        'int2token': dataset.int2token,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'neg_sampling': args.negative_sampling}, 
                       osp.join(config.model_dir, args.task_name, f'model_best.pth'))
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from libcity.data.dataset.downstream_task.eta_dataset import LinearETADataset
from libcity.data.dataset.downstream_task.similarity_dataset import SimilarityDataset
from libcity.executor.downstream_task.eta_executor import ETAExecutor
from libcity.executor.downstream_task.sim_executor import SimExecutor
from libcity.model.downstream_task.linear_eta import LinearETA
from libcity.model.downstream_task.linear_sim import LinearSim
from libcity.utils import set_random_seed, ensure_dir
from libcity.utils.downstream_utils import get_logger
import torch


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="train.py")

    parser.add_argument("--data", default="./raw_data",
                        help="Path to training and validating data")

    parser.add_argument("--dataset", type=str, default="cd",
                        help="Choose the dataset:bj/xa/cd")

    parser.add_argument("--use_cache", type=str2bool, default=True,
                        help="Choose to use cache")

    parser.add_argument("--seed", type=int, default=0,
                        help="seed")

    parser.add_argument("--plot_eta", type=str2bool, default=True,
                        help="Choose to plot ETA result")

    parser.add_argument("--roadmap", type=str, default="",
                        help="Choose the dataset: beijing/porto")

    parser.add_argument("--checkpoint_name", type=str, default="",
                        help="Choose the dataset: beijing/porto")

    parser.add_argument("--exp_id", type=int, default=-1,
                        help="exp_id")

    parser.add_argument("--emb_id", type=int, default=-1,
                        help="emb_id")

    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of workers for dataloader")

    parser.add_argument("--dim", type=int, default=128,
                        help="The hidden state size in the transformer encoder")

    parser.add_argument("--learner", type=str, default="adamw",
                        help="")
    parser.add_argument("--lr_scheduler", type=str, default="cosinelr",
                        help="")
    parser.add_argument("--embedding_size", type=int, default=128,
                        help="The word (cell) embedding size")

    parser.add_argument("--p_dropout", type=float, default=0.5,
                        help="The dropout probability")

    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size")

    parser.add_argument("--max_epoch_sim", type=int, default=50,
                        help="The number of training epochs")
    parser.add_argument("--model_name", type=str, default='HHGCL',
                        help="upstream model name")
    parser.add_argument("--task_name", type=str, default='ETA+Sim',
                        help="task name")
    parser.add_argument("--max_epoch_eta", type=int, default=50,
                        help="The number of training epochs")

    parser.add_argument("--cuda", type=str2bool, default=True,
                        help="True if we use GPU to train the model")

    parser.add_argument("--load_pretrain", type=str2bool, default=True,
                        help="True if we use GPU to train the model")

    parser.add_argument("--gpu_id", type=int, default=0,
                        help="True if we use GPU to train the model")


    parser.add_argument("--max_len", type=int,default=64,
            help="The maximum length of the trajectory sequence")


    parser.add_argument("--embedding_name", type=str, default='',
                        help="road_embedding")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    set_random_seed(args.seed)
    # Key args
    if args.exp_id == -1:
        args.exp_id = int(random.SystemRandom().random() * 1000000)
    args.data_path = os.path.join('./raw_data/', args.dataset)  # './raw_data/bj''
    args.data_cache_path = f'./libcity/cache/'
    args.cache_dir = f'./libcity/cache/{args.exp_id}'
    ensure_dir(args.data_cache_path)
    ensure_dir(args.cache_dir)
    args.roadmap = 'roadmap_{}'.format(args.dataset)
    logger = get_logger(args.exp_id, args.model_name, args.task_name, args.dataset, args.cache_dir)
    logger.info(str(args))
    cuda_condition = torch.cuda.is_available() and args.cuda
    args.device = torch.device(f"cuda:{args.gpu_id}" if cuda_condition else "cpu")
    args.train_data_path = os.path.join(args.data_path, f"traj_road_{args.dataset}_11_train.csv")
    args.eval_data_path = os.path.join(args.data_path, f"traj_road_{args.dataset}_11_val.csv")
    args.test_data_path = os.path.join(args.data_path, f"traj_road_{args.dataset}_11_test.csv")
    args.geo_path = os.path.join(args.data_path, args.roadmap, f"{args.roadmap}.geo")
    args.rel_path = os.path.join(args.data_path, args.roadmap, f"{args.roadmap}.rel")
    print(args.train_data_path)
    embedding_path = 'libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'.format(
        args.emb_id, 'HOME', args.dataset, args.dim
    )
    #eta
    task_model = LinearETA(args=args)
    executor = ETAExecutor(args=args, model=task_model)
    train_dataset = LinearETADataset(args, args.train_data_path, embedding_path=embedding_path)
    eval_dataset = LinearETADataset(args, args.eval_data_path, embedding_path=embedding_path)
    test_dataset = LinearETADataset(args, args.test_data_path, embedding_path=embedding_path)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                   shuffle=True)
    eval_data_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                  shuffle=True)
    executor.train(train_dataloader=train_data_loader, eval_dataloader=eval_data_loader,
                   test_dataloader=test_data_loader)
    executor.evaluate(test_data_loader)
    #sim
    train_set_path = 'raw_data/{}/'.format(args.dataset) + f'train_path_10000.csv'
    test_set_path = 'raw_data/{}/'.format(args.dataset) + f'test_path_5000.csv'
    val_set_path = 'raw_data/{}/'.format(args.dataset) + f'val_path_5000.csv'
    train_sim_label_path = 'raw_data/{}/'.format(args.dataset) + 'simi_label_train.npy'
    test_sim_label_path = 'raw_data/{}/'.format(args.dataset) + 'simi_label_test.npy'
    val_sim_label_path = 'raw_data/{}/'.format(args.dataset) + 'simi_label_val.npy'
    train_sim_label = np.load(train_sim_label_path)
    test_sim_label = np.load(test_sim_label_path)
    val_sim_label = np.load(val_sim_label_path)
    train_dataset = SimilarityDataset(args, train_set_path, embedding_path, train_sim_label)
    test_dataset = SimilarityDataset(args, test_set_path, embedding_path, test_sim_label)
    val_dataset = SimilarityDataset(args, val_set_path, embedding_path, val_sim_label)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=5000, num_workers=args.n_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    task_model = LinearSim(args)
    executor = SimExecutor(args, task_model, train_sim_label=train_sim_label, val_sim_label=val_sim_label,
                                         test_sim_label=test_sim_label)
    executor.train(train_data_loader, val_data_loader, test_data_loader)
    executor.evaluate(test_data_loader)
    exit()

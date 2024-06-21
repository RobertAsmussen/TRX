import os
from datetime import datetime
import torch
import numpy as np
import argparse
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model import CNN_TRX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random
from ray import train, tune
import ray.cloudpickle as raypickle
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.tensorboard.summary import hparams

class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

seed = 2

#setting up seeds
def set_random_seed(manualSeed):
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

def evaluate_model(model_path, data_dir, num_test_tasks, args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join("HP2_eval", timestamp)
    writer = CorrectedSummaryWriter(log_path)

    gpu_device = 'cuda'
    device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
    model = init_model(args, device)
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    with open(model_path, "rb") as fp:
        checkpoint_state = raypickle.load(fp)
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    video_loader = load_data(args, data_dir)

    accuracy, confidence, test_loss, f1, f1_with_names, used_val_classes = test(model, video_loader, num_test_tasks, device)

    metric_dict = { 
        'test/accuracy': accuracy,
        'test/confidence': confidence,
        'test/loss': test_loss,
        'test/f1_score': f1.mean()
    }

    f1_with_names = {f"f1_score/{key}": value for key, value in f1_with_names.items()}

    metric_dict.update(f1_with_names)

    writer.add_hparams({
        'dataset': args.dataset,
        'lr': args.lr,
        'backbone': args.method,
        'temp_set': str(args.temp_set),
        'sequenz length': args.seq_len,
        'way': args.way,
        'shot': args.shot,
        'tasks per batch': args.tasks_per_batch,
        'optimizer': args.opt,
        'query per class': args.query_per_class,
        'query per class test': args.query_per_class_test,
        'img_size': args.img_size,
        'trans_dropout': args.trans_dropout,
        'num_gpus': args.num_gpus,
        'num_workers': args.num_workers,
        'trans_linear_in_dim': args.trans_linear_in_dim,
        'trans_linear_out_dim': args.trans_linear_out_dim,
        'num_test_tasks': num_test_tasks,
        },
        metric_dict
    )

    writer.close()

def prepare_task(task_dict, device, images_to_device = True):
    context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
    target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
    real_target_labels = task_dict['real_target_labels'][0]
    batch_class_list = task_dict['batch_class_list'][0]
    support_n_frames = task_dict['support_n_frames'][0]
    target_n_frames = task_dict['target_n_frames'][0]

    if images_to_device:
        context_images = context_images.to(device)
        target_images = target_images.to(device)
    context_labels = context_labels.to(device)
    target_labels = target_labels.type(torch.LongTensor).to(device)
    support_n_frames = support_n_frames.to(device)
    target_n_frames = target_n_frames.to(device)

    return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, support_n_frames, target_n_frames

def test(model, video_loader, num_test_task, device):
    model.eval()
    with torch.no_grad():

        video_loader.dataset.train = False
        class_folder = video_loader.dataset.class_folders
        test_loss = 0
        accuracies = []
        used_val_classes = set()
        f1_with_class_names = {}
        all_f1 = [] 
        
        for i, task_dict in enumerate(video_loader):
            if i >= num_test_task:
                break

            used_val_classes.update(task_dict['real_target_labels_names'][0])
            context_images, target_images, context_labels, target_labels, real_target_labels, _, support_n_frames ,target_n_frames = prepare_task(task_dict, device)
            model_dict = model(context_images, context_labels, target_images, support_n_frames, target_n_frames)
            target_logits = model_dict['logits']

            averaged_predictions = torch.logsumexp(target_logits, dim=0)
            prediction = torch.argmax(averaged_predictions, dim=-1)

            f1_task = multiclass_f1_score(prediction, target_labels, num_classes=args.way, average=None)
            for it, f1 in enumerate(f1_task):
                all_f1.append(f1)
                class_name = (task_dict['real_target_labels_names'][it])[0]
                if f1_with_class_names.get(class_name) is None:
                    f1_with_class_names[class_name] = []
                f1_with_class_names[class_name].append(f1.item())
        
            accuracy = torch.mean(torch.eq(target_labels, prediction).float())
            accuracies.append(accuracy.item())
            test_loss += loss(target_logits, target_labels, device)

            del target_logits

        accuracy = np.array(accuracies).mean() * 100.0
        confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        f1_with_class_names = {key: np.array(values).mean() for key, values in f1_with_class_names.items()}
        f1 = torch.tensor(all_f1).mean()
        video_loader.dataset.train = True
    
    test_loss = test_loss/num_test_task
    model.train()
    print(f"{model_path} tested")

    return accuracy, confidence, test_loss, f1, f1_with_class_names, used_val_classes

def init_model(args, device):
    model = CNN_TRX(args).to(device)
    
    return model

def load_data(args, data_dir):
    vd = video_reader.VideoDataset(args)
    class_folder = vd.class_folders
    video_loader = torch.utils.data.DataLoader(vd, batch_size=1, num_workers=args.num_workers)
    
    return video_loader

def ray_config_to_args(data_dir, config_path):
    result = train.Result.from_path(config_path)
    config = result.config
    args = ArgsObject(data_dir, config)
    
    return args

class ArgsObject(object):
    def __init__(self, data_dir, config):
        self.path = os.path.join(data_dir, "data", "surgicalphasev1_Xx256")
        self.traintestlist = os.path.join(data_dir, "splits", "surgicalphasev1TrainTestlist")
        self.lr = config["lr"]
        self.dataset = config["dataset"]
        self.split = config["split"]
        self.way = config["way"]
        self.shot = config["shot"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.seq_len = config["seq_len"]
        self.img_size = config["img_size"]
        self.temp_set = config["temp_set"]
        self.debug_loader = False
        self.trans_dropout = config["trans_dropout"]
        self.method = config["method"]
        self.num_gpus = config["num_gpus"]
        self.num_workers = config["num_workers"]
        self.opt = config["Optimizer"]
        self.tasks_per_batch = config["tasks_per_batch"] 
        
        if config["method"] == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
            
        self.trans_linear_out_dim = config["trans_linear_out_dim"]

if __name__ == "__main__":
    data_dir = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets"
    num_test_tasks = 10000
    model_path_list = [
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00004_4_lr=0.0001,query_per_class=2,temp_set=2_2024-06-12_00-00-41/checkpoint_000091/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00005_5_lr=0.0000,query_per_class=2,temp_set=2_2024-06-12_00-00-41/checkpoint_000091/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00007_7_lr=0.0001,query_per_class=3,temp_set=2_2024-06-12_00-00-41/checkpoint_000043/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00011_11_lr=0.0000,query_per_class=4,temp_set=2_2024-06-12_00-00-41/checkpoint_000075/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00013_13_lr=0.0001,query_per_class=5,temp_set=2_2024-06-12_00-00-41/checkpoint_000097/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0b53e_00028_28_lr=0.0001,query_per_class=5,temp_set=2_3_2024-06-12_00-00-41/checkpoint_000030/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0e38d_00007_7_lr=0.0001,query_per_class=3,temp_set=2_2024-06-16_19-58-38/checkpoint_000015/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0e38d_00017_17_lr=0.0000,query_per_class=2,temp_set=2_3_2024-06-16_19-58-38/checkpoint_000077/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0e38d_00020_20_lr=0.0000,query_per_class=3,temp_set=2_3_2024-06-16_19-58-38/checkpoint_000068/data.pkl",
        "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results/tune_with_parameters_0e38d_00022_22_lr=0.0001,query_per_class=4,temp_set=2_3_2024-06-16_19-58-38/checkpoint_000049/data.pkl"
    ]

    for model_path in model_path_list:
        config_path = os.path.dirname(model_path)
        config_path = os.path.dirname(config_path)
        args = ray_config_to_args(data_dir, config_path)
        evaluate_model(model_path, data_dir, num_test_tasks, args)
import os
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
    writer = SummaryWriter(comment="HP2_eval_")
    temp_set_string = args.temp_set
    args.temp_set = [int(t) for t in str.split(temp_set_string)]

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

    accuracy, confidence, test_loss, f1, used_val_classes = test(model, video_loader, num_test_tasks)

    writer.add_hparams({
        'dataset': args.dataset,
        'lr': args.lr,
        'backbone': args.method,
        'temp_set': temp_set_string,
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
        'num_test_tasks': num_test_tasks
        },
        {'test/accuracy': accuracy,
        'test/confidence': confidence,
        'test/test_loss': test_loss,
        'test/F1_Score': f1
        }
    )

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
        class_folder = video_loader.dataset.class_folder
        test_loss = 0
        accuracies = []
        used_val_classes = set()
        all_predictions = []
        all_target_labels = []

        for i, task_dict in enumerate(video_loader):
            if i >= num_test_task:
                break

            used_val_classes.update(task_dict['real_target_labels_names'][0])
            context_images, target_images, context_labels, target_labels, real_target_labels, _, support_n_frames ,target_n_frames = prepare_task(task_dict, device)
            model_dict = model(context_images, context_labels, target_images, support_n_frames, target_n_frames)
            target_logits = model_dict['logits']

            averaged_predictions = torch.logsumexp(target_logits, dim=0)
            prediction = torch.argmax(averaged_predictions, dim=-1)

            real_label_prediction = torch.tensor([real_target_labels[i] for i in prediction])
            all_predictions.extend(real_label_prediction)
            all_target_labels.append(target_labels)

            accuracy = torch.mean(torch.eq(target_labels, prediction).float())
            accuracies.append(accuracy.item())
            test_loss += loss(target_logits, target_labels, device)

            del target_logits

        accuracy = np.array(accuracies).mean() * 100.0
        confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        f1 = multiclass_f1_score(torch.tensor(all_predictions), torch.tensor(all_target_labels), len(set(all_target_labels)), average=None)
        f1_with_class_names = {name: value for name, value in zip(class_folder, f1)}
        print(f1_with_class_names)
        video_loader.dataset.train = True
    
    test_loss = test_loss/num_test_task
    model.train()

    return accuracy, confidence, test_loss, f1, used_val_classes

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

class ArgsObject(object):
    def __init__(self, data_dir, config):
        self.path = os.path.join(data_dir, "data", "surgicalphasev1_Xx256")
        self.traintestlist = os.path.join(data_dir, "splits", "surgicalphasev1TrainTestlist")
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
        
        if config["method"] == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
            
        self.trans_linear_out_dim = config["trans_linear_out_dim"]

if __name__ == "__main__":
    model_path = ""
    data_dir = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets"
    config_path = ""
    num_test_tasks = 100
    args = ray_config_to_args(data_dir, config_path)

    evaluate_model(model_path, data_dir, num_test_tasks, args)
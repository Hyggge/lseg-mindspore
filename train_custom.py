import os
import time
import random
import torch
import torch.nn as nn
import torch.cuda.amp as amp # add mixed precision
import torchvision.transforms as transforms

from argparse import ArgumentParser
from options import Options
from modules.lsegmentation_module import LSegmentationModule
from modules.models.lseg_net import LSegNet
from data import get_dataset
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from data import get_dataset, get_available_datasets
from encoding_custom.nn.loss import SegmentationLosses
from encoding_custom.utils.metrics import SegmentationMetric
from encoding_custom.utils.metrics import batch_pix_accuracy, batch_intersection_union

norm_mean= [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
up_args = {'mode': 'bilinear', 'align_corners': True}
mixed_precision = False
log_dir = ""

def create_log_dir():
    global log_dir
    current_time = datetime.now()
    log_dir = current_time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    log_dir = os.path.join("./logs", log_dir)
    os.makedirs(log_dir)


def get_train_dataloader(dataset_name, data_path, base_size, crop_size, batch_size, augment=False):
    mode =  "train_x" if augment == True else "train"

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    train_dataset = get_dataset(
        dataset_name,
        root=data_path,
        split="train",
        mode=mode,
        transform=train_transform,
        base_size=base_size,
        crop_size=crop_size
    )

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        worker_init_fn=lambda x: random.seed(time.time() + x),
    )


def get_val_dataloader(dataset_name, data_path, base_size, crop_size, batch_size, augment=False):
    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    mode = "val_x" if augment == True else "val"

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    val_dataset = get_dataset(
        dataset_name,
        root=data_path,
        split="val",
        mode=mode,
        transform=val_transform,
        base_size=base_size,
        crop_size=crop_size
    )

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    
def get_labels(dataset):
    labels = []
    path = 'label_files/{}_objectInfo150.txt'.format(dataset)
    assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
    f = open(path, 'r') 
    lines = f.readlines()      
    for line in lines: 
        label = line.strip().split(',')[-1].split(';')[0]
        labels.append(label)
    f.close()
    if dataset in ['ade20k']:
        labels = labels[1:]
    return labels

def get_optimizer(model, base_lr, max_epochs, midasproto, weight_decay):
    params_list = [
        {"params":model.pretrained.parameters(), "lr": base_lr},
    ]
    if hasattr(model, "scratch"):
        print("Found output scratch")
        params_list.append(
            {"params": model.scratch.parameters(), "lr": base_lr * 10}
        )
    if hasattr(model, "auxlayer"):
        print("Found auxlayer")
        params_list.append(
            {"params": model.auxlayer.parameters(), "lr": base_lr * 10}
        )
    if hasattr(model, "scale_inv_conv"):
        print(model.scale_inv_conv)
        print("Found scaleinv layers")
        params_list.append(
            {
                "params": model.scale_inv_conv.parameters(),
                "lr": base_lr * 10,
            }
        )
        params_list.append(
            {"params": model.scale2_conv.parameters(), "lr": base_lr * 10}
        )
        params_list.append(
            {"params": model.scale3_conv.parameters(), "lr": base_lr * 10}
        )
        params_list.append(
            {"params": model.scale4_conv.parameters(), "lr": base_lr * 10}
        )

    if midasproto:
        print("Using midas optimization protocol")
        
        opt = torch.optim.Adam(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / max_epochs, 0.9)
        )

    else:
        opt = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / max_epochs, 0.9)
        )

    return opt, sch

def _filter_invalid(self, pred, target):
    valid = target != self.other_kwargs["ignore_index"]
    _, mx = torch.max(pred, dim=1)
    return mx[valid], target[valid]


def train(model, dataloader, criterion, optimizer, epoch, accumulate_grad_batches=1):
    # set model to train mode
    model.train()
    
    # init
    sample_num = 0
    train_loss_total = 0

    # start training
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for i, batch in loop:
        # get input data
        img, target = batch
        img = img.to(device)
        target = target.to(device)
        # forward
        with amp.autocast(enabled=mixed_precision):
            out = model(img)
            multi_loss = isinstance(out, tuple)
            if multi_loss:
                loss = criterion(*out, target)
            else:
                loss = criterion(out, target)
            loss = amp.GradScaler(enabled=mixed_precision).scale(loss)
            loss = loss / accumulate_grad_batches

        # backward
        if (i+1) % accumulate_grad_batches == 0:
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss.backward(retain_graph=True)
        # update metrics
        # final_output = out[0] if multi_loss else out
        # train_pred, train_gt = _filter_invalid(final_output, target)
        # if train_gt.nelement() != 0:
        #     self.train_accuracy(train_pred, train_gt)
        sample_num += 1
        train_loss_total += loss.item()

        # Show progress while training
        loop.set_description(f'Epoch {epoch + 1}')
        loop.set_postfix(batch_loss=loss.item(), acc="todo", avg_loss=f"{train_loss_total / sample_num:.4f}")

        
    
    train_loss_avg = train_loss_total / sample_num
    
    return {
        "train_loss_avg": train_loss_avg
    }


def val(model, dataloader, criterion, metric):
    # set model to eval mode
    model.eval()

    # init
    sample_num = 0
    val_loss_total = 0
    metric.reset()

    # start evalation
    with torch.no_grad():
        loop= tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for i, batch in loop:
            # get input data
            img, target = batch
            img = img.to(device)
            target = target.to(device)

            # forward
            out = model(img)  
            multi_loss = isinstance(out, tuple)
            if multi_loss:
                val_loss = criterion(*out, target)
            else:
                val_loss = criterion(out, target)
            
            # update metrics
            final_output = out[0] if multi_loss else out
            # valid_pred, valid_gt = _filter_invalid(final_output, target)
            metric.update(target, final_output)
            sample_num += 1
            val_loss_total += val_loss.item()

            # Show progress while evalating
            loop.set_description(f'Evalating')

        pixAcc, iou = metric.get()
        val_loss_avg = val_loss_total / sample_num
        metric.reset()

        return {
            "pixAcc": pixAcc,
            "iou": iou,
            "val_loss_avg": val_loss_avg
        }

if __name__ == "__main__":
    # create log dir
    create_log_dir()
    # get the arguments
    args = Options().parser.parse_args()
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the dataloaders
    if args.dataset == "citys":
        base_size = 2048
        crop_size = 768
    else:
        base_size = 520
        crop_size = 480

    train_dataloader = get_train_dataloader(
        args.dataset, 
        args.data_path, 
        base_size, 
        crop_size, 
        args.batch_size, 
        args.augment
    )

    val_dataloader = get_val_dataloader(
        args.dataset, 
        args.data_path, 
        base_size, 
        crop_size, 
        args.batch_size, 
        args.augment
    )
    
    # get the labels
    labels = get_labels(args.dataset)

    # get the criterion
    criterion = SegmentationLosses(
        se_loss=args.se_loss, 
        aux=args.aux, 
        nclass=len(labels), 
        se_weight=args.se_weight, 
        aux_weight=args.aux_weight, 
        ignore_index=args.ignore_index
    )

    # get the model
    model = LSegNet(
        labels=labels,
        backbone=args.backbone,
        features=args.num_features,
        crop_size=crop_size,
        arch_option=args.arch_option,
        block_depth=args.block_depth,
        activation=args.activation,
    )

    model.pretrained.model.patch_embed.img_size = (crop_size, crop_size)
    model.to(device)

    # get the optimizer and scheduler
    optimizer, scheduler = get_optimizer(
        model, 
        args.base_lr / 16 * args.batch_size,
        args.max_epochs, 
        args.midasproto, 
        args.weight_decay
    )

    # get the metric
    metric = SegmentationMetric(len(labels))


    # train start
    for epoch in range(args.max_epochs):
        # train
        train_result_epoch = train(model, train_dataloader, criterion, optimizer, epoch,  args.accumulate_grad_batches)
        print(f"Train loss: {train_result_epoch['train_loss_avg']:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(log_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # validate
        val_result_epoch = val(model, val_dataloader, criterion, metric)
        print(f"Validation loss: {val_result_epoch['val_loss_avg']:.4f}, \
                pixAcc: {val_result_epoch['pixAcc']:.4f}, \
                iou: {val_result_epoch['iou']:.4f}")



import os
import mindspore as ms
import mindspore.nn as nn
import msadapter.pytorch as torch
import msadapter.torchvision.transforms as transforms

from modules.models.lseg_net import LSegNet
from fcn import FCN8s
from data import get_dataset
from tqdm import tqdm
from datetime import datetime
from options import Options
from encoding_custom.nn.loss import SegmentationLosses
from msadapter.pytorch.utils.data import DataLoader
from encoding_custom.utils.metrics import SegmentationMetric
from mindspore.common.tensor import Tensor

norm_mean= [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
up_args = {'mode': 'bilinear', 'align_corners': True}
log_dir = ""

def create_log_dir():
    global log_dir
    current_time = datetime.now()
    log_dir = current_time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("./log"):
        os.makedirs("./log")
    log_dir = os.path.join("./log", log_dir)
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

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
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

    return DataLoader(
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

class DynamicDecayLR(nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, lr, step_per_epoch, max_epochs):
        super(DynamicDecayLR, self).__init__()
        self.lr = lr
        self.step_per_epoch = step_per_epoch
        self.max_epochs = max_epochs

    def construct(self, global_step):
        current_epoch = global_step // self.step_per_epoch
        return self.lr * pow(1.0 - current_epoch / self.max_epochs, 0.9)

def get_optimizer(model, base_lr, step_per_epoch, max_epochs, midasproto, weight_decay):
    params_list = [
        {"params":model.pretrained.parameters(), "lr": DynamicDecayLR(base_lr, step_per_epoch, max_epochs)},
    ]
    if hasattr(model, "scratch"):
        print("Found output scratch")
        params_list.append(
            {"params": model.scratch.parameters(), "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs)}
        )
    if hasattr(model, "auxlayer"):
        print("Found auxlayer")
        params_list.append(
            {"params": model.auxlayer.parameters(), "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs)}
        )
    if hasattr(model, "scale_inv_conv"):
        print(model.scale_inv_conv)
        print("Found scaleinv layers")
        params_list.append(
            {
                "params": model.scale_inv_conv.parameters(),
                "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs),
            }
        )
        params_list.append(
            {"params": model.scale2_conv.parameters(), "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs)}
        )
        params_list.append(
            {"params": model.scale3_conv.parameters(), "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs)}
        )
        params_list.append(
            {"params": model.scale4_conv.parameters(), "lr": DynamicDecayLR(base_lr * 10, step_per_epoch, max_epochs)}
        )

    if midasproto:
        print("Using midas optimization protocol")
        opt = nn.Adam(
            params_list,
            learning_rate=DynamicDecayLR(base_lr, step_per_epoch, max_epochs),
            weight_decay=weight_decay,
        )

    else:
        opt = nn.Momentum(
            params_list,
            learning_rate=DynamicDecayLR(base_lr, step_per_epoch, max_epochs),
            momentum=0.9,
            weight_decay=weight_decay,
        )
    return opt

def _filter_invalid(self, pred, target):
    valid = target != self.other_kwargs["ignore_index"]
    _, mx = torch.max(pred, dim=1)
    return mx[valid], target[valid]


def train(model, dataloader, criterion, optimizer, epoch, accumulate_grad_batches=1):
    # init
    sample_num = 0
    train_loss_total = 0

    # start training
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    model_with_loss = nn.WithLossCell(model, criterion)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for i, batch in loop:
        # get input data
        img, target = batch
        target = Tensor(target, ms.int32)
        print(img.shape)
        # forward
        loss = train_step(img, target)
        loss = loss / accumulate_grad_batches

        sample_num += 1
        train_loss_total += loss.item()

        # Show progress while training
        loop.set_description(f'Epoch {epoch + 1}')
        loop.set_postfix(batch_loss=loss.item(), acc="todo", avg_loss=f"{train_loss_total / sample_num:.4f}")

    res = {
        "train_loss_avg": train_loss_total / sample_num
    }
    
    return res


def val(model, dataloader, criterion, metric):
    # init
    sample_num = 0
    val_loss_total = 0
    metric.reset()

    # start evalation
    loop= tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    model_with_eval = nn.WithEvalCell(model, criterion)

    for i, batch in loop:
        # get input data
        img, target = batch
        target = Tensor(target, ms.int32)

        # forward
        val_loss, out, target = model_with_eval(img, target)

        sample_num += 1
        val_loss_total += val_loss.item()
        metric.update(target, out)

        # Show progress while evalating
        loop.set_description(f'Evalating')

    pixAcc, iou = metric.get()
    res = {
        "val_loss_avg": val_loss_total / sample_num,
        "pixAcc": pixAcc,
        "iou": iou,
    }

    return res


if __name__ == "__main__":
    # create log dir
    create_log_dir()

    # get the arguments
    args = Options().parser.parse_args()

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
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # get the model
    # model = FCN8s(n_class=len(labels)) # only for test
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

    # get the optimizer and scheduler
    # optimizer = nn.Adam(
    #     model.trainable_params(),
    #     learning_rate=DynamicDecayLR(
    #         args.base_lr / 16 * args.batch_size,
    #         len(train_dataloader) // (args.batch_size * args.accumulate_grad_batches),
    #         args.max_epochs
    #     ),
    # ) # only for test
    optimizer = get_optimizer(
        model, 
        args.base_lr / 16 * args.batch_size,
        len(train_dataloader) // (args.batch_size * args.accumulate_grad_batches),
        args.max_epochs, 
        args.midasproto, 
        args.weight_decay
    )

    print(f"Optimizer is {optimizer}")

    # get the metric
    metric = SegmentationMetric(nclass=len(labels))

    for epoch in range(args.max_epochs):
        # train
        train_result_epoch = train(model, train_dataloader, criterion, optimizer, epoch,  args.accumulate_grad_batches)
        print(f"Train loss: {train_result_epoch['train_loss_avg']:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(log_dir, f"epoch_{epoch+1}.pth")
        ms.save_checkpoint(model, ckpt_path)

        # validate
        val_result_epoch = val(model, val_dataloader, criterion, metric)
        print(f"Validation loss: {val_result_epoch['val_loss_avg']:.4f}, \
                pixAcc: {val_result_epoch['pixAcc']:.4f}, \
                iou: {val_result_epoch['iou']:.4f}")

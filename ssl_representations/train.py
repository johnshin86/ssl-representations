r"""PyTorch Remote Sensing Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""

import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection

from engine import train_one_epoch
import augment
import vicreg
import utils

from torchvision.models import resnet18
import torchvision.transforms as T

def get_voc(root_path: str, image_set: str, transforms):

    t = []
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    #the VOCDetection class will handle the pathing
    dataset = torchvision.datasets.VOCDetection(root = root_path, transform=transforms, image_set = image_set)

    return dataset

def get_dataset(name: str, image_set: str, transform, data_path: str):
    paths = {
        "voc": (data_path, get_voc),
    }
    p, ds_fn = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds


def get_transform(train, data_preprocess):
    return augment.VOC_preprocess(data_preprocess) if train else None


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='SSL training for representation an analysis', add_help=add_help)

    parser.add_argument('--data-path', default='/media/john/EEA Drive 1/datasets/VOC2012/', help='dataset')
    parser.add_argument('--dataset', default='voc', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="cosineannealinglr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/media/john/EEA Drive 1/ssl_representations/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--framework', default="vicreg", help='SSL framework to utilize (default: vicreg)')
    parser.add_argument('--data-preprocess', default="normalize", help='data preprocess policy (default: normalize)')
    parser.add_argument('--data-augment', default="vicreg", help='data augment policy (default: vicreg)')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    print("Data augmentation type:", args.data_preprocess)

    dataset = get_dataset(args.dataset, "train", get_transform(True, args.data_preprocess),
                                       args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    
    # need to chop off linear classifier of existing library models 
    # and add an expander function

    # TODO: handle model import and expander better. 

    if args.model == "resnet18":
        model = resnet18(pretrained = False)

        # rewrite fc layer as expander
        model.fc = torch.nn.Sequential(torch.nn.Linear(512, 2048),
                    torch.nn.BatchNorm1d(num_features = 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 2048),
                    torch.nn.BatchNorm1d(num_features = 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 2048),                     
                     )

    print(model)

    model.to(device)

    # convert batchnorms for distributed environment
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]

    loss_dict = {}
    loss_coeff = {}

    if args.framework == 'vicreg':
        loss_dict["mse"] = torch.nn.MSELoss(reduction='mean')
        loss_dict["var"] = vicreg.VarianceLoss()
        loss_dict["cov"] = vicreg.CovarianceLoss()

        loss_coeff["lambda"] = 25.0
        loss_coeff["mu"] = 25.0
        loss_coeff["nu"] = 1.0
    else:
        raise ValueError(f'Unimplemented SSL framework"{args.framework}"')

    augment_policy = None

    if args.data_augment == "vicreg":
        augment_policy = augment.VOC_augment(args.data_augment)
    else:
        raise ValueError(f'Unimplemented augment policy"{args.data_preprocess}"')

    # add options for optimizer
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, loss_dict, loss_coeff, augment_policy)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch,
                'loss_coeff': loss_coeff
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
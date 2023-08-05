import logging
import pathlib
import random
import shutil
import time
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from dataset import VolData
from model import *
from model_ocuc import *
import torchvision
from torch import nn
from torch import optim
from tqdm import tqdm
from torchvision.utils import make_grid
from losses import *
from dataset import VolData
import h5py 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):
    
    train_data = VolData(root =args.train_path)
    validation_data = VolData(root=args.validation_path)

    return validation_data, train_data


def create_data_loaders(args):
    
    validation_data, train_data = create_datasets(args)   
#     display_data = [validation_data[i] for i in range(0, len(validation_data), len(validation_data) // 4)]
    display_data = [validation_data[i] for i in range(len(validation_data)//2, len(validation_data), 2)]


    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True,
    )
    validation_loader = DataLoader(
        dataset=validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        shuffle = True,
        num_workers=16,
        #pin_memory=True,
    )
    return train_loader, validation_loader, display_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):
        
        input, target = torch.unsqueeze(data[0],dim=1), torch.unsqueeze(data[1],dim=1)
        input, target = input.cuda(), target.cuda()
        output = model(input.float())
        output = F.sigmoid(output)

        loss_ce = nn.BCELoss()(output, target.float())
        loss_seg_dice = dice_loss(output, target)
        loss = loss_ce+loss_seg_dice  #loss if only CE and DL 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Train_loss_ce', loss_ce, global_step + iter)
        writer.add_scalar('Train_loss_dice', loss_seg_dice, global_step + iter)
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


#         if iter % args.report_interval == 0:
        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss = {loss.item():.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
        )
        start_iter = time.perf_counter()

    return loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    ce_losses = []
    dice_losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            input, target = torch.unsqueeze(data[0],dim=1), torch.unsqueeze(data[1],dim=1)
            input, target = input.cuda(), target.cuda()
            output = model(input.float())
            output = F.sigmoid(output)

            loss_ce = nn.BCELoss()(output, target.float())
            loss_seg_dice = dice_loss(output, target)

            loss = loss_ce+loss_seg_dice
            losses.append(loss.item())
            ce_losses.append(loss_ce.item())
            dice_losses.append(loss_seg_dice.item())

            # filename = str(args.preds_dir) + f"{epoch}.h5" 
            # with h5py.File(filename,'w') as data:
            #     data.create_dataset('input',data = input.cpu())
            #     data.create_dataset('target_seg',data = target.cpu()) 
            #     data.create_dataset('predicted_seg',data = output.cpu())
       
        writer.add_scalar('validation_Loss',np.mean(losses),epoch)
        writer.add_scalar('validation_Loss_ce', np.mean(ce_losses),epoch)
        writer.add_scalar('validation_Loss_dice', np.mean(dice_losses),epoch)

    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        grid = torchvision.utils.make_grid(image)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):  
            input, target = torch.unsqueeze(data[0],dim=1), torch.unsqueeze(data[1],dim=1)
            input, target = input.cuda(), target.cuda()
            output = model(input.float())
            output = F.sigmoid(output)
            output[output<0.5]=0.0
            output[output>=0.5]=1.
#             print(f"min:{torch.min(target)},max:{torch.max(target)}")

            print("input: ", torch.min(input), torch.max(input))
            print("target: ", torch.min(target), torch.max(target))
            print("predicted: ", torch.min(output), torch.max(output))

            save_image(input, 'Input')
            save_image(target, 'Target')
            
            save_image(output, 'Predicted')

            save_image(torch.abs(target.float() - output.float()), 'Error')
            filename = str(args.preds_dir) +"/epoch_" + f"{epoch}.h5" 
            with h5py.File(filename,'w') as data:
                data.create_dataset('input',data = input.cpu())
                data.create_dataset('target_seg',data = target.cpu()) 
                data.create_dataset('predicted_seg',data = output.cpu())
            break

def save_model(args, exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_validation_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        
        
def build_model(args):
#     model = kiunet().to(args.device)
    model = UNet().to(args.device)
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 

def build_optim(args, params):
#     optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
#     optimizer = torch.optim.Adadelta(params, args.lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay, foreach=None,maximize=False)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer



def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 1
        best_validation_loss= checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)   
        optimizer = build_optim(args, model.parameters())
        best_validation_loss = 1e9 #Inital validation loss
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, validation_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer)
        validation_loss,validation_time = evaluate(args, epoch, model, validation_loader, writer)
        visualize(args, epoch, model, display_loader, writer)
        scheduler.step(epoch)

        is_new_best = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss,validation_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'validationLoss= {validation_loss:.4g} TrainTime = {train_time:.4f}s validationTime = {validation_time:.4f}s',
        )
    writer.close()
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for CT Segmentation U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=1, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--preds-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where validation predicted images should be saved')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')
    parser.add_argument('--dataset-type',type=str,help='dataset type')
    
    return parser
    
    

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print (args)
    main(args)
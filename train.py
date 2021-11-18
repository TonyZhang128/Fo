import os 
import time
import shutil
import torch
from torch import nn, optim
from torch.cuda import check_error
from torch.utils.tensorboard import SummaryWriter

from core.dataloaders import DataLoaderFactory
from core.models import ModelFactory
from core.dataloaders.youtube_dataset import YoutubeDataset
from core.criterion import SmoothCrossEntropyLoss
from core.optimizer import CustomSchedule
from core.metrics import compute_epiano_accuracy
from options.train_options import TrainOptions


# class Engine(BaseEngine):
# class Engine:

#     def __init__(self, cfg: ConfigTree):
#         self.cfg = cfg
#         print(cfg)
#         self.summary_writer = SummaryWriter()
#         self.model_builder = ModelFactory(cfg)
#         self.dataset_builder = DataLoaderFactory(cfg)

#         self.train_ds = self.dataset_builder.build(split='train')
#         self.test_ds = self.dataset_builder.build(split='val')
#         self.ds: YoutubeDataset = self.train_ds.dataset

#         self.train_criterion = nn.CrossEntropyLoss(
#             ignore_index=self.ds.PAD_IDX
#         )
#         self.val_criterion = nn.CrossEntropyLoss(
#             ignore_index=self.ds.PAD_IDX
#         )
#         self.model: nn.Module = self.model_builder.build(device=torch.device('cuda'), wrapper=nn.DataParallel)
#         optimizer = optim.Adam(self.model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
#         self.optimizer = CustomSchedule(
#             self.cfg.get_int('model.emb_dim'),
#             optimizer=optimizer,
#         )

#         self.num_epochs = cfg.get_int('num_epochs')

#         # logger.info(f'Use control: {self.ds.use_control}')

#     def train(self, epoch=0):
#         loss_meter = AverageMeter('Loss')
#         acc_meter = AverageMeter('Acc')
#         num_iters = len(self.train_ds)
#         self.model.train()
#         for i, data in enumerate(self.train_ds):
#             midi_x, midi_y = data['midi_x'], data['midi_y']

#             if self.ds.use_pose:
#                 feat = data['pose']
#             elif self.ds.use_rgb:
#                 feat = data['rgb']
#             elif self.ds.use_flow:
#                 feat = data['flow']
#             else:
#                 raise Exception('No feature!')

#             feat, midi_x, midi_y = (
#                 feat.cuda(non_blocking=True),
#                 midi_x.cuda(non_blocking=True),
#                 midi_y.cuda(non_blocking=True)
#             )

#             if self.ds.use_control:
#                 control = data['control']
#                 control = control.cuda(non_blocking=True)
#             else:
#                 control = None

#             output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

#             loss = self.train_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

#             self.optimizer.zero_grad()
#             loss.backward()

#             self.optimizer.step()

#             acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.PAD_IDX)

#             batch_size = len(midi_x)
#             loss_meter.update(loss.item(), batch_size)
#             acc_meter.update(acc.item(), batch_size)

#             # logger.info(
#             #     f'Train [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
#             #     f'{loss_meter}\t{acc_meter}'
#             # )
#         self.summary_writer.add_scalar('train/loss', loss_meter.avg, epoch)
#         self.summary_writer.add_scalar('train/acc', acc_meter.avg, epoch)
#         return loss_meter.avg

#     def test(self, epoch=0):
#         loss_meter = AverageMeter('Loss')
#         acc_meter = AverageMeter('Acc')
#         num_iters = len(self.test_ds)
#         self.model.eval()

#         with torch.no_grad():
#             for i, data in enumerate(self.test_ds):
#                 midi_x, midi_y = data['midi_x'], data['midi_y']

#                 if self.ds.use_pose:
#                     feat = data['pose']
#                 elif self.ds.use_rgb:
#                     feat = data['rgb']
#                 elif self.ds.use_flow:
#                     feat = data['flow']
#                 else:
#                     raise Exception('No feature!')

#                 feat, midi_x, midi_y = (
#                     feat.cuda(non_blocking=True),
#                     midi_x.cuda(non_blocking=True),
#                     midi_y.cuda(non_blocking=True)
#                 )

#                 if self.ds.use_control:
#                     control = data['control']
#                     control = control.cuda(non_blocking=True)
#                 else:
#                     control = None

#                 output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

#                 """
#                 For CrossEntropy
#                 output: [B, T, D] -> [BT, D]
#                 target: [B, T] -> [BT]
#                 """
#                 loss = self.val_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

#                 acc = compute_epiano_accuracy(output, midi_y)

#                 batch_size = len(midi_x)
#                 loss_meter.update(loss.item(), batch_size)
#                 acc_meter.update(acc.item(), batch_size)
#                 # logger.info(
#                 #     f'Val [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
#                 #     f'{loss_meter}\t{acc_meter}'
#                 # )
#             self.summary_writer.add_scalar('val/loss', loss_meter.avg, epoch)
#             self.summary_writer.add_scalar('val/acc', acc_meter.avg, epoch)

#         return loss_meter.avg

#     @staticmethod
#     def epoch_time(start_time: float, end_time: float):
#         elapsed_time = end_time - start_time
#         elapsed_mins = int(elapsed_time / 60)
#         elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#         return elapsed_mins, elapsed_secs

#     def run(self):
#         best_loss = float('inf')
#         for epoch in range(self.num_epochs):
#             start_time = time.time()
#             _train_loss = self.train(epoch)
#             loss = self.test(epoch)
#             end_time = time.time()
#             epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

#             # logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')

#             is_best = loss < best_loss
#             best_loss = min(loss, best_loss)
#             save_checkpoint(
#                 {
#                     'state_dict': self.model.module.state_dict(),
#                     'optimizer': self.optimizer.state_dict()
#                 },
#                 is_best=is_best,
#                 folder=experiment_path
#             )

#     def close(self):
#         self.summary_writer.close()
# die

class AverageMeter:
    '''
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L354
    '''

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(opt, epoch, train_ds, ds, summary_writer, model, optimizer, train_criterion):
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    model.train()
    num_iters = len(train_ds)
    for i, data in enumerate(train_ds):
        # midi_x, midi_y = data['midi_x'], data['midi_y']
        midi = data[1]
        # if ds.use_pose:
        feat = data[0]
        # elif ds.use_rgb:
        #     feat = data['rgb']
        # elif ds.use_flow:
        #     feat = data['flow']
        # else:
        #     raise Exception('No feature!')

        feat, midi = (
            feat.cuda(non_blocking=True),
            midi.cuda(non_blocking=True),
        )

        ds.use_control = None
        if ds.use_control:
            control = data['control']
            control = control.cuda(non_blocking=True)
        else:
            control = None

        output = model(pose=feat, tgt=midi, pad_idx=ds.PAD_IDX, control=control)
        loss = train_criterion(output.view(-1, output.shape[-1]), midi.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = compute_epiano_accuracy(output, midi, pad_idx=ds.PAD_IDX)
        batch_size = len(midi)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)

        # logger.info(
        #     f'Train [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
        #     f'{loss_meter}\t{acc_meter}'
        # )
        print(f'Train [{epoch}]/{opt.num_epoch}][{i}/{num_iters}]\t')
        print(f'{loss_meter}\t{acc_meter}')

    summary_writer.add_scalar('train/loss', loss_meter.avg, epoch)
    summary_writer.add_scalar('train/acc', acc_meter.avg, epoch)
    return loss_meter.avg

def val(opt, epoch, val_ds, ds, summary_writer, model, optimizer, val_criterion):
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    num_iters = len(val_ds)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_ds):
            # midi_x, midi_y = data['midi_x'], data['midi_y']
            midi = data[1]

            # if ds.use_pose:
            feat = data[0]
            # elif ds.use_rgb:
            #     feat = data['rgb']
            # elif ds.use_flow:
            #     feat = data['flow']
            # else:
            #     raise Exception('No feature!')

            feat, midi = (
                feat.cuda(non_blocking=True),
                midi.cuda(non_blocking=True)
            )
            ds.use_control = None
            if ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
            else:
                control = None

            try: output = model(pose=feat, tgt=midi, pad_idx=ds.PAD_IDX, control=control)
            except: 
                print(data)

            """
            For CrossEntropy
            output: [B, T, D] -> [BT, D]
            target: [B, T] -> [BT]
            """
            loss = val_criterion(output.view(-1, output.shape[-1]), midi.flatten())

            acc = compute_epiano_accuracy(output, midi)

            batch_size = len(midi)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            # logger.info(
            #     f'Val [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
            #     f'{loss_meter}\t{acc_meter}'
            # )
            print(f'Val [{epoch}]/{opt.num_epoch}][{i}/{num_iters}]\t')
            print(f'{loss_meter}\t{acc_meter}')
        summary_writer.add_scalar('val/loss', loss_meter.avg, epoch)
        summary_writer.add_scalar('val/acc', acc_meter.avg, epoch)

    return loss_meter.avg

def save_checkpoint(state_dict, is_best=False, folder='', filename='ckpt.pth', best_loss=0):
    filename = os.path.join(folder, 'loss_%.4f_' % best_loss + filename)
    if is_best:
        torch.save(state_dict, filename)
        shutil.copyfile(filename, os.path.join(folder, 'best', 'loss_%.4f_' % best_loss+'best.pth'))

def main():
    #parse arguments
    opt = TrainOptions().parse()
    opt.device = torch.device("cuda")

    #construct data loader
    dataset_builder = DataLoaderFactory(opt)
    train_ds = dataset_builder.build(split='train')    
    ds: YoutubeDataset = train_ds.dataset

    #create validation set data loader if validation_on option is set
    if opt.validation_on:
        # opt.mode = 'val'
        val_ds = dataset_builder.build(split='val')

    # Tensorboard start
    summary_writer = SummaryWriter()

    # Network Builders
    model_builder = ModelFactory(opt)
    model = model_builder.build(device = opt.device)
    checkpoint_path = "/data/zyn/Foley_extracted/ckpt/best/loss_3.0825_best.pth"
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    # Set up loss functions
    train_criterion = nn.CrossEntropyLoss(
        ignore_index = ds.PAD_IDX
    )
    val_criterion = nn.CrossEntropyLoss(
        ignore_index = ds.PAD_IDX
    )

    num_epoch = opt.num_epoch
    best_loss = float('inf')
    for epoch in range(num_epoch):
        start_time = time.time()
        train_loss = train(opt, epoch, train_ds, ds, summary_writer, model, optimizer, train_criterion)
        val_loss = val(opt, epoch, val_ds, ds, summary_writer, model, optimizer, val_criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s') 

        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)
        save_checkpoint(
            state_dict = model.module.state_dict(),
            is_best=is_best,
            folder=opt.ckpt,
            best_loss = best_loss
        )

if __name__ == '__main__':
    main()

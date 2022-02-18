import yaml
import os
import numpy as np
from tqdm import tqdm
import time
import math
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
# from dataloaders import make_data_loader
from dataloaders.utils import  *
from dataloaders.custom_dataset import CustomDataset
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from dataloaders import custom_transforms as tr
from PIL import Image


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, setting=None):
        super(DeepLab, self).__init__()
        if isinstance(setting, str):
            with open(setting, 'r') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            self.yaml = setting

        self.root_path = self.yaml['DATASET']['root_path']
        self.image_path = self.yaml['DATASET']['image_path']
        self.annot_path = self.yaml['DATASET']['annot_path']
        self.save_dir = os.path.join(self.yaml['file_path'], self.yaml['train']['exp'])
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        self.num_classes = num_classes

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn
        self.best_pred = 0.0
        self.evaluator = Evaluator(self.num_classes)

        self.is_cuda = torch.cuda.is_available()
        self._device = torch.device('cuda') if self.is_cuda else torch.device('cpu')

        self.best_valid = float("inf")
        
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    # @TODO
    def fit(self, x, y, validation_data, epochs=150, batch_size=8):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda else torch.device('cpu')
        self._device = _device
        
        if batch_size == 1:
            print("batch size must more than 1. batch size is overrided 4.")
            batch_size = 4

        cfg = self.yaml['train']

        trainpack = (x, y)
        validpack = validation_data

        trainset = CustomDataset(trainpack, setting=self.yaml, mode='train')
        validset = CustomDataset(validpack, setting=self.yaml, mode='valid')

        num_class = trainset.num_classes if trainset.num_classes == self.num_classes else None
        if num_class is None:
            print("train dataset nc {} is different with model nc {}.".format(trainset.num_classes, self.num_classes))
            exit(0)

        kwargs = {'num_workers': cfg['workers'], 'pin_memory': True, 'drop_last': True}
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = DataLoader(validset, batch_size=1, shuffle=True, **kwargs)

        total_train_iter = math.ceil(len(x) / batch_size)
        total_valid_iter = math.ceil(len(validation_data[0]) / batch_size)

        train_params = [{'params': self.get_1x_lr_params(), 'lr': float(cfg['lr'])},
                        {'params': self.get_10x_lr_params(), 'lr': float(cfg['lr']) * 10}]

        # Define Optimizer
        self.optimizer = torch.optim.SGD(train_params, 
                                         momentum=float(cfg['momentum']),
                                         weight_decay=float(cfg['weight_decay']), 
                                         nesterov=float(cfg['nesterov']))

        # Define Criterion
        # whether to use class balanced weights
        weight = None
        # if cfg['use_balanced_weights']:
        #     weight = calculate_weigths_labels(args.dataset, train_loader, self.nclass)
        #     classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
        #     if os.path.isfile(classes_weights_path):
        #         weight = np.load(classes_weights_path)
        #     else:
        #         weight = torch.from_numpy(weight.astype(np.float32))

        self.criterion = SegmentationLosses(weight=weight, cuda=torch.cuda.is_available()).build_loss(mode=cfg['loss_type'])
        self.scheduler = LR_Scheduler(cfg['lr_scheduler'], cfg['lr'],
                                      epochs, len(train_loader))
        
        if torch.cuda.is_available():
            self = self.to(_device)
        
        for epoch in range(0, epochs):
            print(f"\n[Epoch {epoch + 1}/{epochs}] Start")
            epoch_start = time.time() 
            train_loss = 0.0
            self.train()
            num_img_tr = len(train_loader)
            last_iter = 0

            for i, sample in enumerate(train_loader):
                iter_start = time.time()
                self.optimizer.zero_grad()

                image, target = sample['image'], sample['label']
                if torch.cuda.is_available():
                    image, target = image.to(_device), target.to(_device)

                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                output = self(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                last_iter = i + 1
                # tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

                # Show 10 * 3 inference results each epoch
                # if i % (num_img_tr // 10) == 0:
                # global_step = i + num_img_tr * epoch
                if len(train_loader) - (batch_size * (i + 1)) == 1:     # image 1장이 들어오면 오류남.
                    print("only one image -> passed.")
                    break
                print("[train %s/%3s] Epoch: %3s | Time: %6.2fs/it | train_loss: %6.4f" % (
                            i + 1, total_train_iter, epoch + 1, time.time() - iter_start, round((train_loss / last_iter), 4)))
                # self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

            train_loss = round((train_loss / last_iter), 4)
            print("\n[Epoch {} training Ended] > Time: {:.2}s/epoch | Loss: {:.4f}\n".format(
                            epoch + 1, time.time() - epoch_start, train_loss))
            
            if not cfg['no_val']:
                valid_result = self.evaluate(model=self, 
                                             dataloader=valid_loader, 
                                             criterion=self.criterion, 
                                             evaluator=self.evaluator,
                                             cfg=self.yaml['train'])
                valid_loss, valid_Acc, valid_mIoU = valid_result
                valid_loss = round(valid_loss, 4)
                valid_Acc = round(valid_Acc, 4)
            else:
                valid_loss, valid_Acc, valid_mIoU = float("-inf"), float("-inf"), float("-inf")

            if valid_loss < self.best_valid:
                torch.save(self, os.path.join(self.save_dir, "best.pt"))
            torch.save(self, os.path.join(self.save_dir, "last.pt"))

    def evaluate(self, model, dataloader, criterion, evaluator, cfg=None):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda else torch.device('cpu')
        model = model.to(_device)

        model.eval()
        evaluator.reset()
        # tbar = tqdm(self.val_loader, desc='\r')
        valid_loss = 0.0
        last_iter = 0
        epoch_start = time.time()
        for i, sample in enumerate(dataloader):
            iter_start = time.time()
            image, target = sample['image'], sample['label']
            if torch.cuda.is_available():
                image, target = image.to(_device), target.to(_device)
            # print(image)
            with torch.no_grad():
                output = model(image)

            loss = criterion(output, target)
            valid_loss += loss.item()
            # tbar.set_description('Test loss: %.3f' % (valid_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            
            evaluator.add_batch(target, pred)
            last_iter = i + 1
            print("[valid %s] | Time: %6.2fs/it | valid_loss: %6.4f" % (
                i + 1, time.time() - iter_start, round((valid_loss / last_iter), 4)))

        valid_loss = (valid_loss / last_iter)
        valid_loss = round(valid_loss, 4)

        print("\n[validation Ended] > Time: {:.2}s/all | Loss: {:.4f}\n".format(
            time.time() - epoch_start, valid_loss))

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        # self.writer.add_scalar('val/total_loss_epoch', valid_loss, epoch)
        # self.writer.add_scalar('val/mIoU', mIoU, epoch)
        # self.writer.add_scalar('val/Acc', Acc, epoch)
        # self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        # self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        table = [['Accuracy', round(Acc, 4)], ['Accuracy per Class', round(Acc_class, 4)], ['mean IoU', round(mIoU, 4)], ['Freq. Weighted IoU', round(FWIoU, 4)]]
        print(tabulate(table, tablefmt="grid"))
        # print('Loss: %.3f' % valid_loss)

        return (valid_loss, Acc, mIoU)

    def predict(self, test_images, use_cpu=False):
        cmap = self.get_colormap(256).tolist()
        palette = [value for color in cmap for value in color]

        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda and not use_cpu else torch.device('cpu')
        self = self.to(_device)

        cfg = self.yaml['test']
        cfg['sync_bn'] = None if cfg['sync_bn'] == 'None' else cfg['sync_bn']
        cfg['out_path'] = os.path.join(os.getenv("NW_HOME"), f"neusite/jobs/{self.job.job_id}/SegmentedImages/")
        os.makedirs(cfg['out_path'], exist_ok=True)
        self = self.eval()
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        pred_images = []

        with torch.no_grad():
            for i, test_image in enumerate(tqdm(test_images, desc=" -> Predicting... ")):
                _imgpath = os.path.abspath(test_image)
                image = Image.open(_imgpath).convert('RGB')
                sample = {'image': image, 'label': image}
                tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

                if _device != torch.device('cpu'):
                    tensor_in = tensor_in.to(_device)
                # print(tensor_in.shape)
                output = self(tensor_in)
                output_np = torch.max(output[:3], 1)[1].detach().cpu().numpy()
                # print("filename :", _imgpath)
                # print("[IMG_PNG]\n", output_np)
                # print("[IMG_PNG] - unique", np.unique(output_np))
                # exit(0)
                grid_image = make_grid(decode_seg_map_sequence(output_np), 3, normalize=False, range=(0, 255))
                savepath = cfg['out_path'] + "segmented_" + os.path.basename(_imgpath)[:-3] + "png"
                grid_image = grid_image.permute(1, 2, 0).detach().cpu().numpy() * 255.0
                grid_image = grid_image.astype('uint8')

                img_png = np.zeros((grid_image.shape[0], grid_image.shape[1]), np.uint8)
                for index, val_col in enumerate(cmap):
                    img_png[np.where(np.all(grid_image == val_col, axis=-1))] = index
                img_png = Image.fromarray(img_png).convert('P')
                img_png.putpalette(palette)
                img_png.save(savepath)

                pred_images.append(savepath)

        return pred_images

    def get_colormap(self, num=256):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """

        def bit_get(val, idx):
            """Gets the bit value.
            Args:
                val: Input value, int or numpy int array.
                idx: Which bit of the input val.
            Returns:
                The "idx"-th bit of input val.
            """
            return (val >> idx) & 1

        colormap = np.zeros((num, 3), dtype=int)
        ind = np.arange(num, dtype=int)

        for shift in reversed(list(range(8))):
            for channel in range(3):
                colormap[:, channel] |= bit_get(ind, channel) << shift
            ind >>= 3

        return colormap


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr


class CustomDataset(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self, datapack, setting, mode='train'):
                 # base_dir=Path.db_root_dir('pascal'),
                 # split='train',
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        # self._base_dir = base_dir
        self.im_ids = []
        self.images = datapack[0]
        self.categories = datapack[1]
        self.args = setting['train']
        self.mode = mode
        self.num_classes = setting['nc']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.mode == 'train':
            return self._transform(sample, mode='train')
        elif self.mode == 'valid':
            return self._transform(sample, mode='valid')

    def _make_img_gt_point_pair(self, index):
        _imgpath = os.path.abspath(self.images[index])
        _targetpath = os.path.abspath(self.categories[index])

        _img = Image.open(_imgpath).convert('RGB')
        _target = Image.open(_targetpath)#.convert('L')

        return _img, _target

    def _transform(self, sample, mode='train'):
        base_size = self.args['base_size']
        crop_size = self.args['crop_size']

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if mode == 'train':
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=base_size, crop_size=crop_size),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=mean, std=std),
                tr.ToTensor()])

        elif mode == 'valid':
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=crop_size),
                tr.Normalize(mean=mean, std=std),
                tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'NeuralSegmentation(mode=' + str(self.mode) + ')'


if __name__ == '__main__':
    print("main doesn't have any works!")
    exit(0)


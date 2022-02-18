import sys
import torch
import os
import yaml
from glob import glob

from modeling.deeplab import *


def get_model(yaml_path : str = './setting.yaml'):
    file_path = os.path.split(sys.modules[__name__].__file__)[0]
    yaml_file = os.path.join(file_path, yaml_path)

    assert os.path.isfile(yaml_file), "There isn't setting.yaml file!"
    with open(yaml_file) as f:
        setting = yaml.load(f, Loader=yaml.SafeLoader)
    assert len(setting), "either setting value must be specified. (yaml file is empty.)"

    if not isinstance(setting['classes'], list) and setting['classes'].split('.')[1] == 'txt':
        file_list = glob(f"{setting['DATASET']['root_path']}/**/{setting['classes']}", recursive=True)
        assert len(file_list) == 1, "Error."
        file_list = file_list[0]
        class_txt = open(file_list, 'r')
        classes = class_txt.readlines()
        class_txt.close()
        for i, c in enumerate(classes):
            if c[-1] == '\n':
                classes[i] = c[:-1]

        setting['classes'] = classes

    setting['nc'] = len(setting['classes'])
    setting['file_path'] = file_path

    setting['train']['sync_bn'], setting['train']['resume'], setting['train']['checkname'] = None, None, None
    setting['train']['cuda'] = not setting['train']['no_cuda'] and torch.cuda.is_available()

    if isinstance(setting['train']['lr'], str) and setting['train']['lr'] != 'None':
        print("learning rate is {}".format(setting['train']['lr']))
        setting['train']['lr'] = float(setting['train']['lr'])

    elif setting['train']['lr'] is None or setting['train']['lr'] == 'None':
        print("learning rate is None state. Override lr: 1e-3.")
        setting['train']['lr'] = 1e-3

    model = DeepLab(num_classes=setting['nc'],
                    backbone=setting['train']['backbone'],
                    output_stride=setting['train']['out_stride'],
                    sync_bn=setting['train']['sync_bn'],
                    freeze_bn=setting['train']['freeze_bn'],
                    setting=setting)

    return model


if __name__ == "__main__":
    setting = yaml.load('setting.yaml')
    model = DeepLab(num_classes=21,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False,
                    setting=setting)

    print(model)
    exit()
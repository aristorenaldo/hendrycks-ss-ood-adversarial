import torch
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import itertools
import random
import logging
import os

import yaml
import json
from types import SimpleNamespace

def shuffle_channel(img: torch.Tensor, index_shuffle: int) -> torch.Tensor:
    """Mengacak urutan dimensi RGB sebagai bentuk transformasi

    Parameters
    ----------
    img : torch.Tensor
        Pixel image RGB

    index_shuffle : int
        Index pengacakan berdasarkan kombinasi RGB
    Returns
    -------
    torch.Tensor
        Shuffled result image
    """
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)

    list_to_permutations = list(itertools.permutations(range(3), 3))
    return img[list_to_permutations[index_shuffle], ...]

class SslTransform(object):
    '''
    Wrapper for SSL Transformation

    Parameter
    ---------
    arch : architecture of Gated-SSL to determine which transformations is used
        (Moe1, Lorot, Moe1Sc, Moe1Flip)
    
    Returns
    -------
    image : torch.Tensor
        Transformed image
    ssl_labels : int | tupple
        SSL label
    '''
    def __init__(self, arch) -> None:
        assert isinstance(arch,str)
        assert arch
        self.arch = arch

    
    def __transform_picker(self, image):
        idx = random.randint(0, 3) # select patch
        idx2 = random.randint(0, 3) # rotation
        r2 = image.size(1)
        r = r2 // 2
        
        if idx == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2

        if self.arch == 'Moe1' or self.arch == 'Nomoe':
            flip_label = random.randint(0, 1)
            sc_label = random.randint(0, 5)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)

            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label, sc_label)

            return image, ssl_label
        
        elif self.arch == 'Lorot':
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            return image, rot_label
        
        elif self.arch == 'Moe1flip':
            flip_label = random.randint(0, 1)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label)
            return image, ssl_label
        
        elif self.arch == 'Moe1sc':
            sc_label = random.randint(0, 5)
             # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, sc_label)
            return image, ssl_label

        raise Exception('arch not implemented')
    
    def __call__(self, image: torch.Tensor):
        assert isinstance(image, torch.Tensor)
        return self.__transform_picker(image)
    
def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
           d = [dict2obj(x) for x in d]
 
    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
           return d
  
    # declaring a class
    class C:
        pass
  
    # constructor of the class passed to obj
    obj = C()
  
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
  
    return obj

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
        # if (k in dct and isinstance(dct[k], dict) ):    
            dict_merge(dct[k], merge_dct[k])
        else:
            if k in dct.keys():
                dct[k] = merge_dct[k]

class Config(object):
    """
    Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """
    def __init__(self, default_path, config_path=None) -> None:
        
        cfg = {}
        if config_path is not None:
            with open(config_path) as cf_file:
                cfg = yaml.safe_load( cf_file.read())     
        
        with open(default_path) as def_cf_file:
            default_cfg = yaml.safe_load( def_cf_file.read())

        dict_merge(default_cfg, cfg)

        self._data = default_cfg
    
    def get(self, path=None, default=None):

        sub_dict = dict(self._data)
        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default
        
class ConfigObj():
    def __init__(self, default_path, config_path=None) -> None:
        cfg = {}
        if config_path is not None:
            with open(config_path) as cf_file:
                cfg = yaml.load( cf_file.read(), Loader=yaml.Loader)     
        
        with open(default_path) as def_cf_file:
            default_cfg = yaml.load( def_cf_file.read(), Loader=yaml.Loader)

        dict_merge(default_cfg, cfg)
        self._data_obj = json.loads(json.dumps(default_cfg), object_hook=lambda item: SimpleNamespace(**item))
    def get(self):
        return self._data_obj
    def __str__(self):
        return str(self._data)

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger

class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
    
    def update(self, tb_dict, epoch, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            epoch: contains information of epoch (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        
        for key, value in tb_dict.items():
            if isinstance(value, dict):
                self.writer.add_scalars(suffix+key, value, epoch)
            else: 
                self.writer.add_scalar(suffix+key, value, epoch) 

if __name__ == "__main__":
    test = ConfigObj('config/test_def.yaml', 'config/test.yaml')
    print(test.get())



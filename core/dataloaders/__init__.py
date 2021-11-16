from pyhocon import ConfigTree
from torch.utils.data import DataLoader

import torch


class DataLoaderFactory:

    # change from other project
    def __init__(self, opt):
        self.opt = opt
        # self.num_gpus = max(1, torch.cuda.device_count())
        self.num_gpus = 1

    def build(self, split='train'):
        # dset = self.cfg.get_string('dataset.dset')
        dset = self.opt.dataset
        if dset == 'URMP':
            from .urmp import URMPDataset
            ds = URMPDataset.from_cfg(self.opt, split=split)
        elif dset == 'urmp_midi2feat':
            from .urmp_midi2feat import URMPMIDI2FeatDataset
            ds = URMPMIDI2FeatDataset.from_cfg(self.opt, split=split)
        elif dset == 'atinpiano':
            from .urmp_music_transformer import URMPDataset
            ds = URMPDataset.from_cfg(self.opt, split=split)
        elif dset == 'youtube_atinpiano':
            from .youtube_dataset import YoutubeDataset
            ds = YoutubeDataset.from_cfg(self.opt, split=split)
        elif dset == 'music21_segment':
            from .youtube_dataset import YoutubeSegmentDataset
            ds = YoutubeSegmentDataset.from_cfg(self.opt, split=split)
        elif dset == 'youtube_urmp':
            from .youtube_dataset import YoutubeURMPDataset
            ds = YoutubeURMPDataset.from_cfg(self.opt, split=split)
        else:
            raise Exception

        loader = DataLoader(
            ds,
            batch_size=self.opt.batchSize * self.num_gpus,
            num_workers=self.opt.nThreads * self.num_gpus,
            shuffle=(split == split)
        )

        print('Real batch size:', loader.batch_size)

        return loader

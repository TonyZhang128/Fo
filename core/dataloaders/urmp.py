from torch.utils import data
from torch.utils.data import Dataset
import random
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np
from core import utils
import torch
from pyhocon import ConfigTree


@dataclass
class Sample:
    midi_path: str
    pose_path: str
    start_time: float
    duration: float


class URMPDataset(Dataset):
    PAD_IDX = 240
    SOS_IDX = 241
    # EOS_IDX = 242

    BODY_PARTS = {
        'body25': 25
    }

    def __init__(
            self,
            opt,
            split_csv_dir: str,
            duration: float,
            fps=29.97,
            events_per_sec=20,
            pose_layout='body25',
            split='train',
            duplication=100
    ):
        self.opt = opt
        self.duration = duration
        self.fps = fps
        self.pose_layout = pose_layout
        self.split_csv_dir = split_csv_dir
        self.duplication = duplication
        self.events_per_sec = events_per_sec

        assert split in ['train', 'val', 'test'], split
        self.split = split
        self.samples = []
        for dataset in self.split_csv_dir:
            if dataset == self.split_csv_dir[0]:
                dataset_name = 'MUSIC'
                # continue
            elif dataset == self.split_csv_dir[1]:
                dataset_name = 'URMP'
            else:
                raise Exception
            for file in os.listdir(dataset):
                csv_path = os.path.join(dataset, file, f'{split}.csv')
                # print(csv_path)
                df = pd.read_csv(str(csv_path))
                midi_dir = os.path.join(self.opt.Foley_extracted, dataset_name, 'midi', file) 
                pose_dir = os.path.join(self.opt.Foley_extracted, dataset_name, 'pose', file) 
                sample = self.build_samples_from_dataframe(self.opt, df, midi_dir, pose_dir, dataset_name)
                self.samples.append(sample)

        # self.samples = self.build_samples_from_dataframe(self.opt, self.df)
        self.samples = [sample for s in self.samples for sample in s]
        # 不同的数据扩充方法
        if split == 'train':
            # for ins_id in range(len(self.samples)):
            #     self.samples[ins_id] = self.samples[ins_id] * duplication
            self.samples *= duplication
        else:
            self.samples = self.split_val_samples_into_small_pieces(self.samples, duration)

        
        self.num_frames = int(duration * fps)
        self.num_events = int(duration * events_per_sec)
        self.body_part = self.BODY_PARTS.get(pose_layout, -1) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample)

        if self.split == 'train':
            start_time = random.random() * (sample.duration - 1.5 * self.duration)
        else:
            start_time = 0.

        # if sample.start_time == None:
        #     print(sample)

        try: start_time += sample.start_time
        except:
            print(sample)

        
        start_frame = int(start_time * self.fps)

        if self.split == 'test':
            # midi = [0] * self.num_events
            # midi[0] = self.SOS_IDX
            # midi[-1] = self.EOS_IDX
            midi = [self.PAD_IDX] * self.num_events
            midi[0] = self.SOS_IDX
        else:
            midi = utils.io.midi_to_list(sample.midi_path, start_time, self.duration)
            midi = self.pad_midi_events(midi)

        pose = utils.io.read_pose_from_npy(sample.pose_path, start_frame, self.num_frames, part=self.body_part)

        midi = torch.tensor(midi)
        pose = torch.from_numpy(pose)

        return pose, midi

    def pad_midi_events(self, midi):
        # new_midi = [self.SOS_IDX] + midi + [self.EOS_IDX]
        new_midi = [self.SOS_IDX] + midi

        if len(new_midi) > self.num_events:
            new_midi = new_midi[:self.num_events]
            # new_midi[-1] = self.EOS_IDX
        elif len(new_midi) < self.num_events: 
            pad = self.num_events - len(new_midi)
            new_midi = new_midi + [self.PAD_IDX] * pad

        return new_midi

    @staticmethod
    def build_samples_from_dataframe(opt, df, midi_dir, pose_dir, dataset_name):
        samples = []
        for _i, row in df.iterrows():
            vid = row.vid
            if dataset_name == 'URMP':
                track_index = row.track_index
                instrument = row.instrument          
                piece = row.piece
                # ScoSep_1_vn_18_Nocturne.mid
                midi_name = 'ScoSep_'+str(track_index)+'_'+instrument+'_'+'%.2d'%vid+'_'+piece+'.mid'
                midi_path = os.path.join(midi_dir, midi_name)
                # PoseSep_1_vn_18_Nocturne.npy
                pose_name = 'PoseSep_'+str(track_index)+'_'+instrument+'_'+'%.2d'%vid+'_'+piece+'.npy'
                pose_path = os.path.join(pose_dir, pose_name)

            elif dataset_name == 'MUSIC':  
                # WVeheJFHQS4.mid 
                midi_name = vid + '.mid'
                midi_path = os.path.join(midi_dir, midi_name)
                # WVeheJFHQS4.npy
                pose_name = vid + '.npy'
                pose_path = os.path.join(pose_dir, pose_name)
            else:
                raise Exception
            sample = Sample(
                midi_path,
                pose_path,
                row.start_time,
                row.duration
            )
            samples.append(sample)
        return samples

    @staticmethod
    def split_val_samples_into_small_pieces(samples, duration: float):
        new_samples = []
        # for ins_samples in samples:
        #     new_ins_samples = []
        for sample in samples:
            stop = sample.duration
            pieces = np.arange(0., stop, duration)[:-1] 
            for new_start in pieces:
                new_samples.append(Sample(
                    midi_path=sample.midi_path,
                    pose_path=sample.pose_path,
                    start_time=new_start,
                    duration=duration,
                ))
        # new_samples.append(new_samples)

        return new_samples

    @classmethod
    def from_cfg(cls, opt, split: str = 'train'):
        # split_csv_dir = cfg.get_string('dataset.split_csv_dir')
        # duration = cfg.get_float('dataset.duration')
        # fps = cfg.get_float('dataset.fps')
        # pose_layout = cfg.get_string('dataset.pose_layout')
        # duplication = cfg.get_int('dataset.duplication')
        # events_per_sec = cfg.get_int('dataset.events_per_sec', 20)
        split_csv_dir = ['./resources/MUSIC']
        split_csv_dir.append(opt.split_csv_dir)
        duration = opt.duration
        fps = opt.fps
        pose_layout = opt.pose_layout
        duplication = opt.duplication
        events_per_sec = opt.events_per_sec
        return cls(
            opt,
            split_csv_dir,
            duration,
            fps=fps,
            pose_layout=pose_layout,
            split=split,
            duplication=duplication,
            events_per_sec=events_per_sec,
        )

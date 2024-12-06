import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


def default_loader(path):
    # 加载多种格式的音频文件，并将其转换为PyTorch张量。
    return torchaudio.load(path, normalization=False)


def default_flist_reader(flist):
    item_list = []
    # defaultdict是Python标准库collections模块中的一个类，它继承自内置的dict类型 ，
    # 但是添加了一个额外的功能：当尝试访问一个不存在的键时，
    # defaultdict会自动提供一个默认值。这个特性对于数据聚合、计数以及初始化复杂的嵌套数据结构特别有用。
    # 创建defaultdict实例需要传入一个函数作为参数 ，这个函数在键不存在时会被调用来生成默认值。
    # 常见的函数如int, list, set等可以用于创建不同类型的默认值。
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


class LibriDataset(Dataset):
    def __init__(
        self,
        opt,
        root,
        flist,
        audio_length=20480,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.opt = opt

        self.file_list, self.speaker_dict = flist_reader(flist)

        self.loader = loader
        self.audio_length = audio_length

        self.mean = -1456218.7500
        self.std = 135303504.0

    def __getitem__(self, index):
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
        audio, samplerate = self.loader(
            os.path.join(self.root, speaker_id, dir_id, "{}.flac".format(filename))
        )

        assert (
            samplerate == 16000
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160

        start_idx = np.random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        audio = audio[:, start_idx : start_idx + self.audio_length]

        # normalize the audio samples
        audio = (audio - self.mean) / self.std
        return audio, filename, speaker_id, start_idx

    def __len__(self):
        return len(self.file_list)

    def get_audio_by_speaker(self, speaker_id, batch_size):
        batch_size = min(len(self.speaker_dict[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[speaker_id][idx]
            )

        return batch

    def get_full_size_test_item(self, index):
        """
        get audio samples that cover the full length of the input files
        used for testing the phone classification performance
        """
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
        audio, samplerate = self.loader(
            os.path.join(self.root, speaker_id, dir_id, "{}.flac".format(filename))
        )

        assert (
            samplerate == 16000
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        ## discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160
        audio = audio[:max_length]

        audio = (audio - self.mean) / self.std

        return audio, filename

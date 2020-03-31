import math
import torch
from torch.nn.utils.rnn import pad_sequence


def select_and_pad_on_presence(features, presence):
    indexes = presence >= 0.5
    lengths = indexes.sum((1, 2)).tolist()
    sequences = features[indexes].split(lengths, 0)
    return pad_sequence(sequences, batch_first=True)


def process_decoded_transform(transform):
    transform[..., :3] = torch.fmod(transform[..., :3], math.pi)  # .chunk(3, -1)
    transform[..., 3:6] = torch.clamp(transform[..., 3:6], 0.5, 1.5)  # .chunk(3, -1)
    transform[..., 6:] = torch.clamp(transform[..., 6:], -1, 1)  # .chunk(3, -1)
    return transform

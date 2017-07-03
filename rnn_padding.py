import torch

'''Utility for Padding Sequences to feed to RNN/LSTM.'''


def pad_single_sequence(single_tensor, length):
    padding_vec_dim = (length - single_tensor.size(0), *single_tensor.size()[1:])
    return torch.cat([single_tensor, torch.zeros(*padding_vec_dim)])


def pad_list_sequences(sequence_list, length=None):
    """Pad Python List of unpadded tensors. If length is 
       not specified padding till sentence of max length."""
    if length is None:
        length = max(s.size(0) for s in sequence_list)
    padded_list = [pad_single_sequence(s, length) for s in
                   sorted(sequence_list, key=lambda tensor: tensor.size(0), reverse=True)]
    # Now add extra dimension
    padded_list = [s.view(length, 1, s.size(1)) for s in padded_list]
    return torch.cat(padded_list, dim=1), list(sorted((s.size(0) for s in sequence_list), reverse=True))

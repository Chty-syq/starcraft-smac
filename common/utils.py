import torch


def batch_np_to_tensor(batch, device):
    for key in batch.keys():
        batch[key] = torch.from_numpy(batch[key]).to(device)
    return batch

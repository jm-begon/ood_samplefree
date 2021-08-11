from torch.utils.data import DataLoader, ConcatDataset, Subset

def get_transform(dataset):
    if isinstance(dataset, DataLoader):
        return get_transform(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return get_transform(dataset.datasets[0])
    if isinstance(dataset, Subset):
        return get_transform(dataset.dataset)
    return dataset.transform
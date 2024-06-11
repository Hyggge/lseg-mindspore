
from encoding_custom.datasets.ade20k import ADE20KSegmentation 

encoding_datasets = {
    "ade20k": ADE20KSegmentation
}


def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys()) 

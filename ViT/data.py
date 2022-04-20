from sklearn.model_selection import train_test_split, KFold
from datasets import Image, load_dataset, load_metric,\
    Dataset, concatenate_datasets
import datasets
import os
import numpy as np
from tqdm import tqdm


def get_paths(args):
    data_path = args.data_path
    first = [os.path.join(data_path, '1', x) for x in os.listdir(data_path + '1')]
    third = [os.path.join(data_path, '3', x) for x in os.listdir(data_path + '3')]
    drova = [os.path.join(data_path, 'drova', x) for x in os.listdir(data_path + 'drova')]
    paths = np.array(first + third + drova)
    return paths


def split_paths(paths, args):
    if 'n_splits' in args:
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        assert args.n_fold in range(1, args.n_splits+1), "n_fold not in n_splits range"
        for i, (train_index, val_index) in enumerate(kf.split(paths)):
            if i + 1 == args.n_fold:
                train_paths, val_paths = paths[train_index], paths[val_index]
        return train_paths, val_paths

    train_paths, val_paths = train_test_split(paths, test_size=0.3, random_state=args.seed)
    return train_paths, val_paths


def build_dataset(args):
    paths = get_paths(args)
    train_paths, val_paths = split_paths(paths, args)
    data = {
        'train': {
            'image': [],
            'image_file_path': [],
            'labels': []
    },
        'test': {
            'image': [],
            'image_file_path': [],
            'labels': []
    }}

    path2label = {'1': 1, 'drova': 0, '3': 2}
    for train_path in tqdm(train_paths):
        data['train']['image'].append(train_path)
        data['train']['image_file_path'].append(train_path)
        data['train']['labels'].append(path2label[train_path.split('/')[-2]])

    for test_path in tqdm(val_paths):
        data['test']['image'].append(test_path)
        data['test']['image_file_path'].append(test_path)
        data['test']['labels'].append(path2label[test_path.split('/')[-2]])

    features = datasets.Features(
        {
            "image": datasets.Value(dtype='string', id=None),
            "image_file_path": datasets.Value(dtype='string', id=None),
            "labels": datasets.ClassLabel(num_classes=3, names=list(range(3)), names_file=None, id=None),
        }
    )

    train_dataset = Dataset.from_dict(data['train'], features=features)
    test_dataset = Dataset.from_dict(data['test'], features=features)
    train_dataset = train_dataset.cast_column("image", Image())
    test_dataset = test_dataset.cast_column("image", Image())

    dataset = load_dataset("superb", "ks")
    dataset["train"] = train_dataset
    dataset["test"] = test_dataset
    dataset["validation"] = test_dataset
    return dataset

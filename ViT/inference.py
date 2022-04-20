from PIL import Image
from tqdm import tqdm
import argparse
from transformers import AutoFeatureExtractor, ViTForImageClassification
import os
import torch
from torchvision import transforms
import csv


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt inference', add_help=False)
    parser.add_argument('--tta', default=1, type=int)
    parser.add_argument('--data_path', default='data/', type=str)
    parser.add_argument('--checkpoint_path', default='google/vit-base-patch16-224-in21k', type=str)
    parser.add_argument('--submit_file', default='submit.csv', type=str)
    parser.add_argument('--output_dir', default='./vit-base-beans-demo-v5', type=str,
                        help='Checkpoints path')
    return parser


class ViTFold:
    def __init__(self, dev_path, labels):
        self.dev_path = dev_path
        self.labels = labels
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.dev_path + '_fold1')
        self.fold1 = ViTForImageClassification.from_pretrained(
            self.dev_path + '_fold1',
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )
        self.fold2 = ViTForImageClassification.from_pretrained(
            self.dev_path + '_fold2',
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )
        self.fold3 = ViTForImageClassification.from_pretrained(
            self.dev_path + '_fold3',
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )
        self.models = [self.fold1, self.fold2, self.fold3]

    def __call__(self, image):
        image = self.feature_extractor(image, return_tensors='pt')['pixel_values']
        preds = [torch.softmax(model(image)[0], dim=1).cpu() for model in self.models]
        output = torch.mean(torch.cat(preds, dim=0), dim=0)
        return torch.unsqueeze(output, dim=0)


def main(args):
    vit_model = ViTFold(args.checkpoint_path, [0, 1, 2])

    test_images = os.listdir(args.data_path)
    labels = {}

    if args.tta:
        tta_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomRotation(270)
        ]

    for test_image in tqdm(test_images):
        orig_image = Image.open(os.path.join(args.data_path, test_image))
        preds = []
        if args.tta:
            for transform in tta_transforms:
                image = transform(orig_image)
                output = vit_model(image)
                preds.append(output)
        output = vit_model(orig_image)
        preds.append(output)
        mn = torch.mean(torch.cat(preds, dim=0), dim=0)
        labels[int(test_image.split('.')[0])] = torch.argmax(mn)
    labels = {k: labels[k] for k in sorted(labels)}

    with open(args.submit_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'class'])
        for idx, label in labels.items():
            label = label.cpu().numpy()
            if label == 2:
                label = 3
            writer.writerow([idx, label])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vit inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

from data import build_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification,\
    TrainingArguments, Trainer
import torch
import argparse
import numpy as np
from datasets import load_metric
from pathlib import Path
import time
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_splits', default=3, type=int)
    parser.add_argument('--n_fold', default=1, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--model_name_or_path', default='google/vit-base-patch16-224-in21k', type=str,
                        help='Model name or path to checkpoint')
    parser.add_argument('--data_path', default='data/', type=str)
    parser.add_argument('--output_dir', default='./vit-base-beans-demo-v5', type=str)
    return parser


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def main(args):
    print(args)
    dataset = build_dataset(args)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name_or_path)

    def transform(batch):
        nonlocal feature_extractor
        inputs = feature_extractor([x for x in batch['image']], return_tensors='pt')
        inputs['labels'] = batch['labels']
        return inputs

    dataset = dataset.with_transform(transform)
    labels = dataset['train'].features['labels'].names

    model = ViTForImageClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=feature_extractor,
    )

    start_time = time.time()
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    metrics = trainer.evaluate(dataset['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

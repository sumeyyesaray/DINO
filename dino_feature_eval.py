"""
Frozen DINOv2 / DINOv3 backbone feature quality evaluation on CIFAR-10.

Extracts features with a frozen pretrained backbone (no fine-tuning) and
scores them with k-NN and a linear probe, so the numbers can be compared
against the fine-tuned ViT-B/16 vs Swin-T results from the earlier paper.

Usage:
    python dino_feature_eval.py --model facebook/dinov2-small
    python dino_feature_eval.py --model facebook/dinov3-vits16-pretrain-lvd1689m
"""

import argparse

import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(model_name: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def build_transform(processor) -> transforms.Compose:
    size = processor.crop_size["height"] if hasattr(processor, "crop_size") else processor.size["shortest_edge"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def build_dataset(transform: transforms.Compose, train: bool) -> torchvision.datasets.CIFAR10:
    return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)


@torch.no_grad()
def extract_features(model, loader: DataLoader, device: torch.device):
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device)
        cls_token = model(pixel_values=images).last_hidden_state[:, 0, :]
        feats.append(cls_token.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def knn_eval(train_feats, train_labels, test_feats, test_labels, k: int) -> float:
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(train_feats, train_labels)
    return accuracy_score(test_labels, clf.predict(test_feats))


def linear_probe_eval(train_feats, train_labels, test_feats, test_labels) -> float:
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(train_feats, train_labels)
    return accuracy_score(test_labels, clf.predict(test_feats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="facebook/dinov2-small",
        help="HF hub id, e.g. facebook/dinov2-small, facebook/dinov2-base, "
             "facebook/dinov3-vits16-pretrain-lvd1689m",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading backbone: {args.model}")
    processor, model = load_backbone(args.model, device)
    transform = build_transform(processor)

    print("Preparing CIFAR-10 train/test sets")
    train_set = build_dataset(transform, train=True)
    test_set = build_dataset(transform, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Extracting frozen features (train split)")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print("Extracting frozen features (test split)")
    test_feats, test_labels = extract_features(model, test_loader, device)

    print(f"Running k-NN (k={args.knn_k})")
    knn_acc = knn_eval(train_feats, train_labels, test_feats, test_labels, k=args.knn_k)

    print("Running linear probe")
    lp_acc = linear_probe_eval(train_feats, train_labels, test_feats, test_labels)

    print("\n=== Results ===")
    print(f"Model:                  {args.model}")
    print(f"k-NN (k={args.knn_k}) accuracy:   {knn_acc * 100:.2f}%")
    print(f"Linear probe accuracy:  {lp_acc * 100:.2f}%")
    print("\nCompare these against the fine-tuned ViT-B/16 vs Swin-T results from the prior paper.")


if __name__ == "__main__":
    main()

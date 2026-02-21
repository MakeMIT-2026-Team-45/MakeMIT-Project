import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from openai_label_mapping import build_binary_label_map
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


TRASH = "trash"
RECYCLING = "recycling"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_taco_data_dir = script_dir / "data" / "TACO" / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MobileNetV3-Large on a binary trash vs recycling task "
            "derived from TACO labels via an LLM mapping."
        )
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=default_taco_data_dir / "annotations.json",
        help="Path to COCO-style TACO annotations JSON (e.g., data/annotations.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_taco_data_dir,
        help="Directory containing TACO images.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=script_dir / "label_cache.json",
        help="Path to cache label->binary category mappings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "checkpoints",
        help="Directory for model checkpoints and metadata.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used for label remapping.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="If set, only train the final binary (1000->2) head.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


@dataclass
class CropSample:
    image_path: Path
    bbox_xywh: Tuple[float, float, float, float]
    binary_label: int


class TacoObjectCropDataset(Dataset):
    def __init__(self, samples: List[CropSample], tfm: transforms.Compose) -> None:
        self.samples = samples
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        x, y, w, h = sample.bbox_xywh
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(image.width, int(x + w))
        bottom = min(image.height, int(y + h))
        if right <= left or bottom <= top:
            crop = image
        else:
            crop = image.crop((left, top, right, bottom))
        return self.tfm(crop), torch.tensor(sample.binary_label, dtype=torch.long)


def stratified_split(
    samples: List[CropSample],
    test_size: float,
    seed: int,
) -> Tuple[List[CropSample], List[CropSample]]:
    by_class: Dict[int, List[CropSample]] = {0: [], 1: []}
    for s in samples:
        by_class[s.binary_label].append(s)

    rng = random.Random(seed)
    train_out: List[CropSample] = []
    test_out: List[CropSample] = []
    for cls, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        n_test = max(1, int(len(cls_samples) * test_size))
        test_out.extend(cls_samples[:n_test])
        train_out.extend(cls_samples[n_test:])
        print(f"class={cls} train={len(cls_samples)-n_test} test={n_test}")
    rng.shuffle(train_out)
    rng.shuffle(test_out)
    return train_out, test_out


class MobileNetBinaryHead(nn.Module):
    """
    Requested architecture:
      MobileNetV3-Large (pretrained, output 1000 logits)
      + Linear(1000, 2) binary classifier head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        self.binary_head = nn.Linear(1000, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_1000 = self.backbone(x)
        return self.binary_head(logits_1000)


def make_samples(
    annotations: dict,
    images_dir: Path,
    label_map: Dict[str, str],
) -> List[CropSample]:
    categories = {c["id"]: c["name"] for c in annotations["categories"]}
    image_by_id = {img["id"]: img for img in annotations["images"]}

    out: List[CropSample] = []
    for ann in annotations["annotations"]:
        cat_name = categories[ann["category_id"]]
        mapped = label_map[cat_name]
        y = 0 if mapped == TRASH else 1
        img_info = image_by_id[ann["image_id"]]
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        out.append(
            CropSample(
                image_path=img_path,
                bbox_xywh=tuple(ann["bbox"]),
                binary_label=y,
            )
        )
    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.annotations_path = args.annotations_path.expanduser().resolve()
    args.images_dir = args.images_dir.expanduser().resolve()
    args.cache_path = args.cache_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.annotations_path.exists():
        raise FileNotFoundError(
            f"annotations file not found: {args.annotations_path}. "
            "If TACO is in a different location, pass --annotations-path explicitly."
        )
    if not args.images_dir.exists():
        raise FileNotFoundError(
            f"images directory not found: {args.images_dir}. "
            "If TACO images are in a different location, pass --images-dir explicitly."
        )

    annotations = load_json(args.annotations_path)
    category_names = [c["name"] for c in annotations["categories"]]

    label_map = build_binary_label_map(
        category_names=category_names,
        cache_path=args.cache_path,
        openai_model=args.openai_model,
    )
    print("binary label map:")
    for k in sorted(label_map):
        print(f"  {k} -> {label_map[k]}")

    samples = make_samples(annotations, args.images_dir, label_map)
    if len(samples) < 10:
        raise RuntimeError("Too few samples found. Check paths for annotations and images.")
    print(f"total object-crop samples: {len(samples)}")

    train_samples, test_samples = stratified_split(samples, args.test_size, args.seed)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = TacoObjectCropDataset(train_samples, train_tfms)
    test_ds = TacoObjectCropDataset(test_samples, eval_tfms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = MobileNetBinaryHead().to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Weight classes inversely proportional to their frequency in the train split.
    train_counts = torch.zeros(2, dtype=torch.float32)
    for sample in train_samples:
        train_counts[sample.binary_label] += 1.0
    if (train_counts == 0).any():
        raise RuntimeError(
            f"Cannot compute class weights because at least one class is missing: {train_counts.tolist()}"
        )
    class_weights = 1.0 / train_counts
    class_weights = class_weights / class_weights.sum() * 2.0
    class_weights = class_weights.to(device)
    print(
        "train class counts and loss weights: "
        f"trash={int(train_counts[0].item())}, recycling={int(train_counts[1].item())}, "
        f"weights={class_weights.detach().cpu().tolist()}"
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = args.output_dir / "mobilenetv3_taco_binary_best.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "label_to_index": {TRASH: 0, RECYCLING: 1},
                    "test_acc": best_acc,
                },
                ckpt_path,
            )
            print(f"saved best checkpoint to {ckpt_path}")

    save_json(args.output_dir / "label_map_used.json", label_map)
    print(f"done. best_test_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()

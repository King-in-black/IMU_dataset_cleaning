import os
import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from sam_med2d import sam_model_registry
from dataloader_augment import NpyDataset
from loss import CombinedLoss
from metrics import SegMetrics, _threshold, _list_tensor
from functools import partial
import matplotlib.pyplot as plt
# Set random seeds for reproducibility
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)

def set_env_threads():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"

set_env_threads()
def custom_collate_fn(batch, device):
    """
    Custom collate function to handle variable-sized fields like `centreline`.
    Args:
        batch: A list of tuples from the dataset's __getitem__ method.
        device: The device to move the data to (e.g., "cuda" or "cpu").
    Returns:
        A batch of data with properly handled variable-sized fields.
    """
    images = torch.stack([item[0] for item in batch]).to(device)  # Move to device
    gt2D = torch.stack([item[1] for item in batch]).to(device)    # Move to device

    # Handle variable-sized centrelines
    max_points = max([item[2].shape[0] for item in batch])  # Find max number of points
    padded_centrelines = torch.zeros((len(batch), max_points, 2), device=device)  # Padding with zeros
    for i, item in enumerate(batch):
        points = item[2].to(device)
        padded_centrelines[i, :points.shape[0], :] = points

    # Handle variable-sized centreline labels
    max_labels = max([item[3].shape[0] for item in batch])  # Find max number of labels
    padded_labels = torch.zeros((len(batch), max_labels), dtype=torch.int32, device=device)
    for i, item in enumerate(batch):
        labels = item[3].to(device)
        padded_labels[i, :labels.shape[0]] = labels

    bboxes = [item[4] for item in batch]  # Keep bboxes as list (can be moved if needed)
    filenames = [item[5] for item in batch]  # Keep filenames as list

    return images, gt2D, padded_centrelines, padded_labels, bboxes, filenames



def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split the dataset into train, val, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
    img_files = sorted(os.listdir(os.path.join(data_dir, "imgs")))
    gt_files = sorted(os.listdir(os.path.join(data_dir, "gts")))
    centreline_files = sorted(os.listdir(os.path.join(data_dir, "centreline")))

    combined_files = list(zip(img_files, gt_files, centreline_files))
    train_val_files, test_files = train_test_split(combined_files, test_size=test_ratio, random_state=random_seed)
    train_files, val_files = train_test_split(train_val_files, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_seed)

    return {"train": train_files, "val": val_files, "test": test_files}


def create_dataloader(dataset_files, data_dir, batch_size, shuffle=True, num_workers=1, augment=False, aug_num=1, complete_centreline=False, device="cuda"):
    dataset = NpyDataset(
        data_root=data_dir,
        augment=augment,
        aug_num=aug_num,
        complete_centreline=complete_centreline,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=partial(custom_collate_fn, device=device),  # Pass device
    )
def freeze_vit(model):
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = False  # 冻结 ViT 的所有参数
    print("ViT (image_encoder) parameters are frozen.")

# 只解冻 ViT 的 adapter
def freeze_vit_except_adapter(model):
    model = model.to('cuda')  # 确保模型权重在 GPU 上
    for name, param in model.image_encoder.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    print("ViT (image_encoder) parameters are frozen except adapter.")
def check_freeze_status(model):
    model = model.to('cuda')
    print("\n[Checking Freeze Status]")
    for name, param in model.image_encoder.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

class CrohnSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, point, point_label, box):
        image_embedding = self.image_encoder(image)
        point_prompt = [point, point_label] if point is not None else None
        box_prompt = None  # Remove box_prompt

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompt, boxes=box_prompt, masks=None
        )
        low_res_masks, predict_iou = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks, predict_iou

def train_model(args):
    """
    Main training loop.
    """
    split_data = split_dataset(args.true_npy_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=2023)

    train_dataloader = create_dataloader(
        split_data["train"],
        args.true_npy_path,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=args.device  # 传递 device 参数
    )

    val_dataloader = create_dataloader(
        split_data["val"],
        args.true_npy_path,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=args.device  # 传递 device 参数
    )
    sam_model = sam_model_registry[args.model_type](args)
    freeze_vit_except_adapter(sam_model)  # 只训练 adapter
    # freeze_vit(sam_model)  # 完全冻结 ViT
    crohnsam_model = CrohnSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        [param for param in crohnsam_model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: max(1 - epoch / args.num_epochs, 0)
    )
    seg_loss = CombinedLoss()
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Train phase
        crohnsam_model.train()
        train_loss = 0
        for image, gt2D, point, point_label, box, _ in tqdm(train_dataloader):
            image = image.to(args.device)
            gt2D = gt2D.to(args.device)
            point = point.to(args.device)  # padded tensor
            point_label = point_label.to(args.device)  # padded tensor

            optimizer.zero_grad()
            preds, iou_pred = crohnsam_model(image, point, point_label, box)
            loss = seg_loss(preds, gt2D, iou_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation phase
        crohnsam_model.eval()
        val_loss = 0
        with torch.no_grad():
            for image, gt2D, point, point_label, box, _ in tqdm(val_dataloader):
                image, gt2D = image.to(args.device), gt2D.to(args.device)
                preds, iou_pred = crohnsam_model(image, point, point_label, box)
                loss = seg_loss(preds, gt2D, iou_pred)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(crohnsam_model.state_dict(), os.path.join(args.work_dir, "best_model.pth"))

        scheduler.step()

    print("Training completed.")
def test_model(args, model, test_dataloader):
    """
    测试阶段，生成模型预测结果并可视化。
    Args:
        args: 参数对象
        model: 已训练好的模型
        test_dataloader: 测试数据的 DataLoader
    """
    model.eval()
    save_dir = "test_visualizations"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for step, (image, gt2D, point, point_label, box, filenames) in enumerate(tqdm(test_dataloader)):
            image, gt2D = image.to(args.device), gt2D.to(args.device)
            preds, iou_pred = model(image, point, point_label, box)

            # 后处理：将预测结果转为二值分割
            preds_binary = (preds.sigmoid() > 0.5).float()

            # 可视化结果
            for i in range(image.size(0)):  # 遍历 batch 内每张图片
                plt.figure(figsize=(12, 4))

                # 输入图像
                plt.subplot(1, 3, 1)
                plt.imshow(image[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
                plt.title("Input Image")
                plt.axis("off")

                # 真实标签
                plt.subplot(1, 3, 2)
                plt.imshow(gt2D[i].squeeze().cpu().numpy(), cmap='gray')
                plt.title("Ground Truth")
                plt.axis("off")

                # 预测结果
                plt.subplot(1, 3, 3)
                plt.imshow(preds_binary[i].squeeze().cpu().numpy(), cmap='gray')
                plt.title("Prediction")
                plt.axis("off")

                # 保存结果
                save_path = os.path.join(save_dir, f"test_step_{step}_sample_{i}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close()
                print(f"Saved visualization: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_npy_path", type=str, default=r"E:\Segmentation_Crohn_Disease\data\test_npz")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--work_dir", type=str, default=r"E:\Segmentation_Crohn_Disease\results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parallel_cnn", action="store_true", help="Enable parallel CNN architecture")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--sam_checkpoint", type=str, default="None")
    parser.add_argument("--reduce_ratio", type=int, default=16,
                        help="adapter_emebeding to embed_dimension//reduce_ratio")
    parser.add_argument("--encoder_adapter", action='store_false', help="use adapter")
    parser.add_argument("--adapter_mlp_ratio", type=float, default=0.15, help="adapter_mlp_ratio")
    # prompt type
    parser.add_argument("--box_prompt", action='store_true', help='Use box prompt')
    parser.add_argument("--point_prompt", action='store_true', help='Use point prompt')
    # other
    parser.add_argument("--use_wandb", action='store_true', help="use wandb to monitor training")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    train_model(args)

if __name__ == "__main__":
    main()


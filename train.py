## Imports
import torch
from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_PATH
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from util.triplet_dataset import TripletDataset
from util.loss import v8DetectionLoss
from model.model import FGFAModel

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)

    batch_idx_list = []
    cls_list = []
    bboxes_list = [] # Stores normalized [cx, cy, w, h]

    for i, targets_per_image in enumerate([b[1] for b in batch]):
        if targets_per_image.numel() == 0:
            continue

        # Extract class IDs and bounding boxes
        class_ids = targets_per_image[:, 0].long() # Class IDs are typically integers
        bboxes_xywh = targets_per_image[:, 1:5]    # Bounding boxes (cx, cy, w, h)

        # Create batch indices for each object in the current image
        # If image 'i' has Ni objects, create a tensor of Ni 'i's
        batch_indices = torch.full((targets_per_image.shape[0],), i, dtype=torch.long)

        batch_idx_list.append(batch_indices)
        cls_list.append(class_ids)
        bboxes_list.append(bboxes_xywh)

    # Concatenate all lists into single tensors for the entire batch
    if not batch_idx_list: # Handle the case where the entire batch has no objects
        return imgs, {
            "batch_idx": torch.empty(0, dtype=torch.long),
            "cls": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4, dtype=torch.float),
        }
    else:
        return imgs, {
            "batch_idx": torch.cat(batch_idx_list, dim=0),
            "cls": torch.cat(cls_list, dim=0),
            "bboxes": torch.cat(bboxes_list, dim=0),
        }

def train():
    # 1) Setup Device, FGFAModel, and Preprocessor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fgfa_model = FGFAModel(yolo_path=r"demo/yolo_ibis_4.pt").to(device)
    fgfa_model.train() # Set the entire FGFAModel to training mode

    # Preprocessor for dataset
    pp = BasePredictor(overrides={
        "model": r"demo/yolo_ibis_4.pt",
        "imgsz": 640,
    })
    pp.setup_model(model=None, verbose=False)
    pp.setup_source("dataset/Triplet_Birds_FGFA.v4i.yolov12/train/images")

    # 2) Build Dataset and DataLoaders
    train_ds = TripletDataset("dataset/Triplet_Birds_FGFA.v4i.yolov12/train", preprocessor=pp)

    train_size = int(0.85 * len(train_ds))
    val_size = len(train_ds) - train_size

    train_dataset, val_dataset = random_split(
        train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 3) Setup Optimizer and Loss Function
    opt = torch.optim.Adam(fgfa_model.parameters(), lr=5e-4)

    hyp = get_cfg(DEFAULT_CFG_PATH)
    if not all(hasattr(hyp, attr) for attr in ['box', 'cls', 'dfl']):
        raise ValueError("Loaded hyperparameters 'hyp' are missing 'box', 'cls', or 'dfl' attributes. "
                             "Please ensure your default.yaml contains these keys under 'Hyperparameters'.")
    
    # initialize v8DetectionLoss with the internal DetectionModel of FGFAModel
    criterion = v8DetectionLoss(fgfa_model.model, hyp=hyp).to(device)

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)

    # Training Parameters and Logging
    best_val_loss = float('inf')
    best_epoch = -1
    epoch_loss = []
    val_epoch_loss = []
    num_epochs = 30
    warmup_epochs = 10
    initial_lr = 5e-4

    for epoch in range(num_epochs):
        # --- Learning Rate Warmup ---
        if epoch < warmup_epochs:
            warmup_lr = initial_lr * ((epoch + 1) / warmup_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Epoch {epoch+1}: Warmup Learning Rate set to {warmup_lr:.6f}")
        elif epoch == warmup_epochs:
            for param_group in opt.param_groups:
                param_group['lr'] = initial_lr
            print(f"Epoch {epoch+1}: Warmup finished. Learning Rate set to {initial_lr:.6f}")

        # --- Training Phase ---
        fgfa_model.train()
        running_train_loss = 0.0
        batch_count = 0

        for imgs_triplet, label_dict in train_loader:
            batch_count += 1
            print(f"\n--- Epoch {epoch+1}, Batch {batch_count} (Train) ---")

            imgs_triplet = imgs_triplet.to(device)
            for k, v in label_dict.items():
                if isinstance(v, torch.Tensor):
                    label_dict[k] = v.to(device)

            preds = fgfa_model(imgs_triplet)
            loss, loss_items = criterion(preds, label_dict)

            print(f"Total Loss: {loss.item():.4f}")
            running_train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fgfa_model.parameters(), max_norm=1.0)
            opt.step()

        avg_train_loss = running_train_loss / len(train_loader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        fgfa_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (imgs_triplet_val, label_dict_val) in enumerate(val_loader):
                imgs_triplet_val = imgs_triplet_val.to(device)
                for k, v in label_dict_val.items():
                    if isinstance(v, torch.Tensor):
                        label_dict_val[k] = v.to(device)

                preds_val = fgfa_model(imgs_triplet_val)
                val_loss, _ = criterion(preds_val, label_dict_val)
                running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_epoch_loss.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': fgfa_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': best_val_loss,
            }, "yolo_fgfa_v16_basev4.pt")
            print(f"Saved best model at Epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        # Update learning rate scheduler only after warm-up
        if epoch >= warmup_epochs - 1:
            scheduler.step()

    print("\nTraining Complete.")
    print(f"Best model saved from Epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    print("Epoch Training Losses:", epoch_loss)
    print("Epoch Validation Losses:", val_epoch_loss)

    # Plotting losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_epoch_loss, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
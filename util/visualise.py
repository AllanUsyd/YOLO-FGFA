import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def visualise_triplet(batch_tensor, titles=None):
    if batch_tensor.dim() == 5:
        batch_tensor = batch_tensor[0]

    if batch_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 frames, got shape {batch_tensor.shape}")

    if titles is None:
        titles = ['Frame -1 (Prev)', 'Frame (Current)', 'Frame +1 (Next)']

    plt.figure(figsize=(12, 4))
    for i in range(3):
        img = TF.to_pil_image(batch_tensor[i].cpu())
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()

def heat_map(feature, title=None):
    outputs = feature.squeeze(0).mean(dim=0)
    plt.imshow(outputs.detach().cpu().numpy(), cmap="viridis")
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if title:
        plt.title(title)

def view(train_ds):
    x, y = train_ds
    print("x.shape =", x.shape)    # expect (3,3,640,640)
    print("y.shape =", y.shape)    # e.g. (1,5) for one box

    # Visualise

    fig, axs = plt.subplots(1,3,figsize=(12,4))
    for i in range(3):
        img = x[i].permute(1,2,0).cpu()  # (H,W,C)
        axs[i].imshow(img.numpy())
        axs[i].set_title(f"Frame {i-1}")
        axs[i].axis("off")
    print("Boxes:", y)
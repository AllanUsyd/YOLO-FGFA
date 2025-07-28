from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TripletDataset(Dataset):
    def __init__(self, root: str, preprocessor):
        """
        root/
          images/
          labels/
        preprocessor: Ultralytics Preprocessor
        """
        self.pp = preprocessor

        img_dir = Path(root) / "images"
        lbl_dir = Path(root) / "labels"

        # group by video prefix
        groups = {}
        for img_path in sorted(img_dir.glob("*.jpg")):
            vid = img_path.name.split("_frame_")[0]
            groups.setdefault(vid, []).append(img_path)

        # build triplets
        self.triples = []
        for frames in groups.values():
            for i in range(1, len(frames)-1):
                f0, f1, f2 = frames[i-1], frames[i], frames[i+1]
                lbl = lbl_dir / f"{f1.stem}.txt"
                if lbl.exists():
                    self.triples.append((f0,f1,f2,lbl))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        f0, f1, f2, lbl = self.triples[idx]
        I0 = Image.open(f0)
        I1 = Image.open(f1)
        I2 = Image.open(f2)

        I0_np = np.array(I0)
        I1_np = np.array(I1)
        I2_np = np.array(I2)

        # this will return a Tensor of shape (3,3,H,W)
        batch_tensor = self.pp.preprocess([I0_np, I1_np, I2_np])

        # load YOLO txt labels for the middle frame
        boxes = []
        if os.path.getsize(lbl) > 0:
            with open(lbl) as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.split())
                    boxes.append([cls, xc, yc, w, h])
                    
        if boxes:
            target = torch.tensor(boxes)  # [Ni x 5]
        else:
            target = torch.empty((0,5))

        return batch_tensor, target

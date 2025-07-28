import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import your CustomYOLO from model.py
from model.model import CustomYOLO
from util.visualise import visualise_triplet 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize your custom YOLO model
    # It will internally load FGFAModel and its components
    model = CustomYOLO(
        yolo_path=r'demo/yolo_ibis_4.pt',
        combined_model_path=r"demo/yolo_fgfa_v13_basev4.pt",
        device=device
    )
    model.eval() # Set the entire wrapper model to eval mode

    print("FGFA-integrated YOLO model loaded successfully for inference!")

    img_store = r"demo/imgs_dir_6"
    frame_store = Path(img_store)
    paths = sorted(frame_store.glob("*.jpg"))
    
    if len(paths) < 3:
        raise ValueError(f"Not enough images in {img_store} for triplet processing. Need at least 3.")

    # Process images in triplets (previous, current, next)
    
    # Loop over the middle frame of each triplet
    for i in range(1, len(paths) - 1): # i is the index of the current (middle) frame
        # Load the triplet of images
        img_prev = np.array(Image.open(str(paths[i-1])))
        img_current = np.array(Image.open(str(paths[i])))
        img_next = np.array(Image.open(str(paths[i+1])))
        
        # Collect them into a list for `predict`
        img_triplet_list = [img_prev, img_current, img_next]

        print(f"\nProcessing triplet with middle frame: {paths[i].name}")

        # Perform prediction. The `predict` method of `CustomYOLO` will handle
        # preprocessing, the custom forward pass, and NMS.
        results = model.predict(img_triplet_list, conf=0.25, iou=0.7) # Pass desired inference params

        # Display results for the current (middle) frame
        for res in results: # `results` will contain one `Results` object for the middle frame
            if res.boxes: # Check if any boxes were detected
                print(f"Detections for {paths[i].name}:")
                for box in res.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.item()
                    cls = box.cls.item()
                    name = res.names[cls]
                    print(f"  - {name} (Conf: {conf:.2f}): {xyxy}")
                res.plot() # Plot detections on the image
                # To show each image with detections immediately:
                # res.show() # This will open a new window for each image

    print("\nAll triplets processed.")
    # If you want to show all plotted images at the end:
    plt.show()
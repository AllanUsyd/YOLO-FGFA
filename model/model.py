import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from model.Feature_Flow import Flow_estimation
from model.Attention import MultiFrameAttention
from PIL import Image
import numpy as np

class CustomForward(DetectionModel):
    def run_up_to(self, x, stop_layer):
        y = [None] * len(self.model)
        for i, m in enumerate(self.model[:stop_layer]):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x
        return x, y

    def continue_forward(self, x, y, start_layer):
        for i, m in enumerate(self.model[start_layer:], start=start_layer):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x
        return x

class FGFAModel(torch.nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        base_yolo = YOLO(yolo_path)
        self._original_detection_model = base_yolo.model
        self.model = base_yolo.model
        self.model.__class__ = CustomForward

        self.model.nc = self._original_detection_model.nc
        self.model.reg_max = self._original_detection_model.model[-1].reg_max
        self.model.stride = self._original_detection_model.stride

        self.flow = Flow_estimation(n_channels=256, sub_batch=3)
        self.mfa = MultiFrameAttention(in_channels=256, out_channels=256, sub_batch_size=3)

    def forward(self, imgs_triplet):
        """
        Args:
            imgs_triplet: Tensor of shape (B, 3, 3, H, W)
                          where B=batch, 3 frames, 3 channels, HxW
        Returns:
            YOLO raw predictions
        """
        B = imgs_triplet.shape[0]
        flat = imgs_triplet.view(3 * B, 3, imgs_triplet.shape[-2], imgs_triplet.shape[-1])
        feats, skips = self.model.run_up_to(flat, stop_layer=5)

        C, Hf, Wf = feats.shape[1:]
        feats = feats.view(3, B, C, Hf, Wf)
        seq = torch.cat([feats[1], feats[0], feats[2]], dim=0)
        _, warped = self.flow(seq)
        f0_w, f1_w, f2_w = warped.view(3, B, C, Hf, Wf)

        mfa_in = torch.cat([f0_w, feats[1], f2_w], dim=0)
        attn_feats, attn_w = self.mfa(mfa_in)
        fused = self.mfa.aggregate(attn_feats, attn_w, frame_index=1)

        skips[4] = fused
        preds = self.model.continue_forward(fused, skips, start_layer=5)
        return preds

class CustomYOLO(FGFAModel):
    def __init__(self, yolo_path, combined_model_path=None, device='cpu'):
        super().__init__(yolo_path)
        self.to(device)

        if combined_model_path:
            checkpoint = torch.load(combined_model_path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict']) 
            print(f"Loaded combined model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")

        self.names = self._original_detection_model.names 
        self.device = device

    @torch.no_grad()
    def predict(self, source, stream=False, **kwargs):
        """
        Custom predict method that uses the FGFAModel's forward pass.
        Args:
            source (list of np.ndarray or list of PIL.Image): List of 3 images (prev, current, next) for a triplet.
                                                              Expected to be one triplet at a time.
            stream (bool): Not used for single triplet prediction, but kept for compatibility.
            **kwargs: Additional arguments for processing, like `imgsz`, `conf`, `iou`, `verbose`, etc.
                      These will be passed to BasePredictor's preprocess/postprocess.
        Returns:
            list of ultralytics.engine.results.Results: Detection results.
        """
        # Ensure input is a list of 3 images
        if not (isinstance(source, list) and len(source) == 3):
            raise ValueError("For CustomYOLO, 'source' must be a list containing 3 image arrays (prev, current, next).")
        
        imgsz = kwargs.get('imgsz', 640)
        kwargs['imgsz'] = imgsz       
        pp = BasePredictor(overrides=kwargs)
        pp.setup_model(model=self._original_detection_model, verbose=False) 
        pp.setup_source(source)
        
        processed_imgs = []
        for img in source:
            processed_img_tensor = pp.preprocess([img]).squeeze(0) # Remove batch dim (1,3,H,W) -> (3,H,W)
            processed_imgs.append(processed_img_tensor)

        # Stack into (1, 3, 3, H, W) for FGFAModel's expected input
        imgs_triplet_tensor = torch.stack(processed_imgs, dim=0).unsqueeze(0).to(self.device)

        # Pass through your custom FGFAModel
        print(f"DEBUG: imgs_triplet_tensor shape before forward: {imgs_triplet_tensor.shape}")
        raw_predictions = self(imgs_triplet_tensor) # FGFAModel's forward returns raw YOLO preds
    
        # Apply NMS
        detections = non_max_suppression(raw_predictions)

        # Post-process results into ultralytics.engine.results.Results objects
        results = []
        print(f"DEBUG: Original middle image shape (source[1].shape): {source[1].shape}")
        for i, det in enumerate(detections): # Should be one `det` for one triplet
            if det is None or det.shape[0] == 0:
                results.append(Results(source[i], path=None, names=self.names, boxes=None))
                continue

            # Scale boxes back to original image size
            print(f"DEBUG: Boxes BEFORE scaling (det[:, :4]):\n{det[:, :4]}")
            
            proc_h, proc_w = imgs_triplet_tensor.shape[-2], imgs_triplet_tensor.shape[-1]
            det[:, :4] = scale_boxes(
                (proc_h, proc_w),               # now just (640, 640)
                det[:, :4],
                source[1].shape[:2]             # original (1080, 1920)
            )
            print(f"DEBUG: Boxes AFTER scaling (det[:, :4]):\n{det[:, :4]}")

            # Create Results object for the current (middle) frame
            # Assuming you want detections for the middle frame (source[1])
            results.append(Results(source[1], path=None, names=self.names, boxes=det)) # path is optional

        return results


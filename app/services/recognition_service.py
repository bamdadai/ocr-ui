from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
import structlog
import re

from app.strhub.data.module_data import SceneTextDataModule
from app.strhub.models.utils import load_from_checkpoint
# --- CHANGE: Updated import path for batchify from the new common module ---
from app.utils.common import batchify

# --- CHANGE: Use structlog for structured logging ---
logger = structlog.get_logger(__name__)


class RecognitionService:
    """
    A specialized service for recognizing text from cropped word images.
    This service has no knowledge of debug modes or how to visualize results.
    """
    def __init__(self, config: dict):
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.min_conf = config['min_conf']
        
        
        # Load the model and its corresponding transforms
        logger.debug("recognition_service.loading_model", checkpoint=config['checkpoint'])
        self.parseq = load_from_checkpoint(str(config['checkpoint'])).eval().to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)
        logger.debug("recognition_service.initialized")

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Preprocesses a single image to be fed into the model."""
        # Convert NumPy array to the format required by the model
        img = Image.fromarray(img)
        return self.img_transform(img)

    def __call__(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Main entry point for the service. Takes a list of word images
        and returns a list of (text, confidence_score) tuples.
        """
        if not crops:
            return []

        # Batch the images using the generic batchify utility
        batches = batchify(crops, self.preprocess, self.batch_size)
        
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                if not batch:
                    continue
                
                logger.debug(
                    "recognition_service.processing_batch",
                    batch_number=batch_idx + 1,
                    batch_size=len(batch)
                )
                
                inputs = torch.stack(batch, dim=0).to(self.device)
                logits = self.parseq(inputs)
                pred = logits.softmax(-1)
                labels, confidences = self.parseq.tokenizer.decode(pred)
                
                processed_confidences = []
                for i, char_confs in enumerate(confidences):
                    # Critical check for empty tensors to prevent errors
                    if not isinstance(char_confs, torch.Tensor) or char_confs.numel() == 0:
                        final_conf = 0.0
                    else:
                        # Calculate confidence using geometric mean for better stability
                        epsilon = 1e-9
                        log_probs = torch.log(char_confs + epsilon).cpu().numpy()
                        final_conf = float(np.exp(np.mean(log_probs)))

                    processed_confidences.append(final_conf)
                    
                    # If confidence is below the threshold, discard the label
                    if final_conf < self.min_conf:
                        labels[i] = ""

                    # Remove standalone 'x' or 'X' characters from individual predictions
                    labels[i] = re.sub(r'\b[xX]\b', '', labels[i])

                all_labels.extend(labels)
                all_confidences.extend(processed_confidences)
        
        return list(zip(all_labels, all_confidences))
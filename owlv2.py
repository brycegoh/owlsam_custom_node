import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import pipeline, SamModel, SamProcessor
import os


class OwlSam:
    CATEGORY = "owlv2"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required":{
                "images": ("IMAGE",),
                "texts": ("STRING", {"default": "window, doorway"}),
                "threshold": ("FLOAT", { "default": 0.2, "min": 0, "max": 1, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "func"

    def func(self, images, texts, threshold):
        comfy_path = os.environ.get('COMFYUI_PATH')
        if comfy_path is None:
            comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        model_path = os.path.abspath(os.path.join(comfy_path, 'models'))

        detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection", device="cuda", cache_dir=model_path)
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base", cache_dir=model_path).to("cuda")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base", cache_dir=model_path)

        # take image from first batch
        image = images[0]
        # tensor to numpy

        image = 255. * image.cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))

        texts = texts.split(",")
        predictions = detector(
            image,
            candidate_labels=texts,
            threshold=threshold
        )

        result_labels = []
        combined_mask = None
        for pred in predictions:
            box = pred["box"]
            score = pred["score"]
            label = pred["label"]
            box = [round(pred["box"]["xmin"], 2), round(pred["box"]["ymin"], 2), 
                round(pred["box"]["xmax"], 2), round(pred["box"]["ymax"], 2)]

            inputs = sam_processor(
                    image,
                    input_boxes=[[[box]]],
                    return_tensors="pt"
                ).to("cuda")

            with torch.no_grad():
                outputs = sam_model(**inputs)

            mask = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0][0][0].numpy()
            mask = mask[np.newaxis, ...]
            result_labels.append((mask, label))

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.logical_or(combined_mask, mask)
        combined_mask = torch.from_numpy(combined_mask).unsqueeze(0).float()
        return (combined_mask,)

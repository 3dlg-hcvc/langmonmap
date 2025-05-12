from typing import List
import torch
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torchvision.transforms as TS


class GroundingDinoDetector:
    def __init__(self,
                 confidence_threshold: float = 0.6
                 ):
        # model_id = "IDEA-Research/grounding-dino-base"
        # self.confidence_threshold = confidence_threshold
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
        self.model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
        self.transform = TS.Compose(
            [
                TS.ToTensor(),
                TS.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes

    def detect(self,
               image: Image
               ):
        
        image_transformed = self.transform(image)
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image_transformed, 
                caption=self.classes[0], 
                box_threshold=0.60, 
                text_threshold=0.60
            )

        res = {}
        res["boxes"] = []
        res["scores"] = []
        res["labels"] = []
        for box, score, label in zip(boxes, logits, phrases):
            res["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
            res["scores"].append(score.item())
            res["labels"].append(label)
        return res
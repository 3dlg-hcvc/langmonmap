# typing
from typing import List, Dict

import numpy as np
# inference
from inference.models import YOLOWorld

# cv2
import cv2

# supervision
import supervision as sv


class YOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes
        self.model.set_classes(classes)

    def detect(self, image: np.ndarray) -> dict:
        if self.classes is None:
            raise ValueError("Classes must be set before detecting")

        results = self.model.infer(image, confidence=self.confidence_threshold)

        preds = {
            "boxes": [],
            "scores": [],
            "labels": []
        }

        for detection in results.predictions:
            cls = detection.class_id
            class_name = detection.class_name

            if class_name in self.classes and detection.confidence > self.confidence_threshold:
                x1 = detection.x - detection.width / 2
                y1 = detection.y - detection.height / 2
                x2 = detection.x + detection.width / 2
                y2 = detection.y + detection.height / 2

                # Check if box is not a point
                if x1 != x2 and y1 != y2:
                    preds["boxes"].append([x1, y1, x2, y2])
                    preds["scores"].append(detection.confidence)
                    preds['labels'].append(class_name)

        return preds

if __name__ == "__main__":
    import supervision as sv
    from PIL import Image
    
    # Test the YOLO World Detector
    detector = YOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["chair"])

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    # Load an image
    image = cv2.imread("/localhome/sraychau/Downloads/hssd_1.png")

    # Detect objects in the image
    preds = detector.detect(image)

    if len(preds['boxes']) > 0:
        detections = sv.Detections(
            xyxy=np.array(preds['boxes']),
            class_id=np.array([0]),
            confidence=np.array(preds['scores'])
        )

        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        image = Image.open("/localhome/sraychau/Downloads/hssd_1.png")
        svimage = np.array(image)
        svimage = bounding_box_annotator.annotate(svimage, detections)
        svimage = label_annotator.annotate(svimage, detections, labels)
        
        sv.plot_image(svimage[:, :, ::-1])
        Image.fromarray(svimage[:, :, :-1]).save("test_detection.jpg")

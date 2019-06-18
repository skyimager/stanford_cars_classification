
import numpy as np
import cv2

from src import models
from src.utils.image import preprocess_image, resize_image
from src.utils.visualization import draw_box
from src.utils.colors import label_color

class DefectDetectionService():
    def __init__(self):
        model_path = 'data/inference.h5'
        self.model =  models.load_model(model_path, backbone_name='resnet50') 
        self.labels_to_names = {0: 'crack', 1: 'wrinkle'}
        self.threshold = 0.50
        
    def predict_label(self, image):
        
        self.draw = image.copy()
        
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image) 
    
        # process image
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        
        self.results = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score >= self.threshold and label in range(0, 2):
                result = self.labels_to_names[label]
                self.results.append((box,score,result))
        
        return self.results
    
    def get_annotated_img(self):
        
        for result in self.results:
            b = result[0].astype(int)
            color = label_color(result[2])
            draw_box(self.draw, b, color=color)
            caption = "{} {:.3f}".format(result[2], result[1])
            cv2.putText(self.draw, caption, (result[0][0], result[0][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 5, cv2.LINE_8)
            
        return self.draw

        
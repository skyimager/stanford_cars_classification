
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import models
from src.utils.image import read_image_bgr, preprocess_image, resize_image
from src.utils.visualization import draw_box
from src.utils.colors import label_color

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,font_scale=2, thickness=2):
    x, y = coordinates[:2]

    cv2.putText(image_array, text, (x + x_offset, y + y_offset),cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def predict_save(file_path, model, save=False, labels_to_names = {0: 'crack', 1: 'wrinkle'}):

    image = read_image_bgr(file_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score >= 0.50 and label in range(0, 2):
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, b, caption)
            print(caption, b)
            # draw_text(box, draw, labels_to_names[label], [255,255,0], 0, -45, 1, 1)
            cv2.putText(draw, caption, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 5, cv2.LINE_8)
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()
            
            filename, _ = os.path.basename(file_path).split(".")
            dirname = os.path.dirname(file_path)
            dest = os.path.join(dirname, filename+"_"+labels_to_names[label]+".jpg")
            
            if save:
                cv2.imwrite(dest, draw)

if __name__ == "__main__":
    
    model = models.load_model('data/inference.h5', backbone_name='resnet50')
    labels_to_names = {0: 'crack', 1: 'wrinkle'}
    im_path = 'data/images/'
    
    for file in os.listdir(im_path):
        file_path = os.path.join(im_path, file)
        predict_save(file_path, model, save=True)
    
    
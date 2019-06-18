
import os, glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from keras.preprocessing.image import img_to_array
import cv2

def xml_parser(path):
    xml_list = []

    for xml_file in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            image_name = root.find('filename').text
            image = cv2.imread(os.path.join(path,image_name))
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)

            value = (root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text,
                     )

            xml_list.append(value) 
            
    return xml_list

def class_map_csv(class_names, class_ids):    
    class_mapping = pd.DataFrame({"class_names":class_names,
                     "class_ids":class_ids})
    
    return class_mapping

def xml2csvs(path, class_names, class_ids):
    xml_list = xml_parser(path)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    df = pd.DataFrame(xml_list, columns=column_name) 
    
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    val_df = df[~msk]
    
    train_df.to_csv(path+'/train_annotations.csv', header=False, index=None)
    val_df.to_csv(path+'/val_annotations.csv', header=False, index=None)
    
    class_mapping = class_map_csv(class_names, class_ids)
    class_mapping.to_csv(path+"/class_mapping.csv",header=False, index=False)
    

if __name__ == "__main__":
    path = "data/retinanet"
    class_names = ["crack", "wrinkle"]
    class_ids = [0,1]
    xml2csvs(path, class_names, class_ids)
    
    

import argparse
import csv
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import os
from PIL import Image
import sys

class VOCDataset:

    def __init__(self, root_dir, dir_type='train', keep_difficult=False, save_dir = '.', label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root_dir)
        self.dir_type = dir_type
        self.save_dir = save_dir
        if self.dir_type == 'test':
            image_sets_file = self.root / "ImageSets/Main/val.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/train.txt"
        
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            # The following are the classes in VOC but for our case
            # we will only keep tvmonitor and comment out the rest
            
            #self.class_names = ('BACKGROUND',
            #'aeroplane', 'bicycle', 'bird', 'boat',
            #'bottle', 'bus', 'car', 'cat', 'chair',
            #'cow', 'diningtable', 'dog', 'horse',
            #'motorbike', 'person', 'pottedplant',
            #'sheep', 'sofa', 'train', 'tvmonitor')
            
            self.class_names = ['tvmonitor']


        self.class_dict = {class_name: class_name for _, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        return image, boxes, labels

    def create_dataset(self):
        header_row = [['ImageID', 'ClassName', 'XMin', 'YMin', 'XMax', 'YMax', 'Confidence']]
        Confidence = 1
        for image_id in self.ids:
            image = self._read_image(image_id)
            height = image.height
            width = image.width
            boxes, labels, is_difficult = self._get_annotation(image_id)
            for i in range(len(labels)):
                current_row = []
                current_row.append(f"{image_id}.jpg")
                current_row.append(labels[i])
                current_row.append(boxes[i][0]/width) # XMin
                current_row.append(boxes[i][1]/height) # YMin
                current_row.append(boxes[i][2]/width) # XMax
                current_row.append(boxes[i][3]/height) # YMax
                current_row.append(Confidence)
                header_row.append(current_row)
            image.save(os.path.join(self.save_dir, f"{self.dir_type}/{image_id}.jpg"))
        
        with open(os.path.join(self.save_dir, f"{self.dir_type}.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(header_row)


    
    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = Image.open(str(image_file))
        return image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract objects from the COCO 2012 Dataset')

    parser.add_argument("--root_dir", type=str,
                        help= "Directory for root VOC2012")
    parser.add_argument("--save_dir", type=str, default= '.',
                        help= "Directory for the relevant data and annotations")
    parser.add_argument("--dir_type", type=str,
                        help= "train or test")
     
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                        format='\n%(asctime)s %(message)s')

    args = parse_args()

    VOCDataset(root_dir=args.root_dir, dir_type=args.dir_type, save_dir=args.save_dir).create_dataset()


#usage python3 voc_extract_objects.py --root_dir ~/Desktop/VOCdevkit/VOC2012/ --save_dir ./root_voc --dir_type train
import json
import sys
import argparse
import pathlib
import logging
import os
from collections import defaultdict
from PIL import Image
import math
import csv

# coco train 2017 breakdown
#{72: 5805, 77: 6434, 73: 4970, 74: 2262, 75: 5703, 76: 2855})

#coco val 2017 breakdown
#{72: 288, 77: 262, 73: 231, 74: 106, 75: 283, 76: 153})

relevant_categories = {72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell_phone'}
count_dict = defaultdict(lambda: 0)

# the coco dataset has image names that start with leading zero's.
# the total number of characters = 12, so if the image id is 1234, we need to append 8 zeros before
# the id.

NUM_LEN = 12
LEAD_CHAR = '0'

def num_digits(n):
    # given an integer returns the number of digits in that integer
    if n > 0:
        return int(math.log10(n))+1
    elif n == 0:
        return 1
    else:
        return int(math.log10(-n))+2 # doesn't count the leading - sign

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract objects from the COCO 2012 Dataset')

    parser.add_argument("--image_dir", type=str,
                        help= "Directory for loading images")
    parser.add_argument("--save_dir", type=str, default= '.',
                        help= "Directory for the relevant data and annotations")
    parser.add_argument("--json_file", type=str,
                        help= "Annotation file")
    parser.add_argument("--dir_type", type=str,
                        help= "train or test")
     
    return parser.parse_args()

def extract(args):

    with open(os.path.join(args.json_file)) as f:
       j = json.load(f)
       annotations = j['annotations']

       header_row = [['ImageID', 'ClassName', 'XMin', 'YMin', 'XMax', 'YMax', 'Confidence']]
       Confidence = 1

       for item in annotations:
           current_row = []
           if item['category_id'] in relevant_categories:
               # convert the image id to COCO name
               image_name = LEAD_CHAR*(NUM_LEN - num_digits(item['image_id'])) + str(item['image_id']) + '.jpg'
               image = Image.open(os.path.join(args.image_dir, image_name))
               height = image.height
               width = image.width

               box = item['bbox']
               label = relevant_categories[item['category_id']]

               XMin = box[0] / width
               YMin = box[1] / height
               XMax = (box[0] + box[2]) / width
               YMax = (box[1] + box[3]) / height

               current_row.append(f"{item['image_id']}.jpg")
               current_row.append(label)
               current_row.append(XMin)
               current_row.append(YMin)
               current_row.append(XMax)
               current_row.append(YMax)
               current_row.append(Confidence)
               header_row.append(current_row)

               image.save(os.path.join(args.save_dir, f"{item['image_id']}.jpg"))

       with open(os.path.join(args.save_dir, f"{args.dir_type}.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(header_row)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                        format='\n%(asctime)s %(message)s')

    args = parse_args()
    extract(args)

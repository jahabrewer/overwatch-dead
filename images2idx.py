#!/usr/bin/env python

# idx file format reference http://yann.lecun.com/exdb/mnist/

import argparse
import os
import glob
import gzip
from PIL import Image

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("image_dir", help="directory containing all images, subdir'd by label")
parser.add_argument("label_file", help="file containing a list of labels, one per line")
parser.add_argument("output_image_file", help="name of gzipped output image IDX file")
parser.add_argument("output_label_file", help="name of gzipped output label IDX file")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

def int_to_ubyte(i):
    assert type(i) is int
    assert i >= 0 and i < 256
    byte_list = []
    byte_list.append(i % 2**8)
    return bytearray(byte_list)

def int_to_bytes(i):
    assert type(i) is int
    byte_list = []
    # TODO this probably isn't portable to different endianness systems
    byte_list.append((i >> 24) % 2**8)
    byte_list.append((i >> 16) % 2**8)
    byte_list.append((i >> 8) % 2**8)
    byte_list.append((i >> 0) % 2**8)
    return bytearray(byte_list)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# read labels
with open(args.label_file) as f:
    lines = f.readlines()
    # strip the newline with rstrip
    labels = [line.rstrip() for line in lines if len(line) > 1]

# print the labels
if args.verbose:
    for i in range(len(labels)):
        print('{0}: {1}'.format(i, labels[i]))


with gzip.open(args.output_label_file, 'wb') as idx_label_file, \
     gzip.open(args.output_image_file, 'wb') as idx_image_file:

        # each subdir represents a label
        subdirs = frozenset(get_immediate_subdirectories(args.image_dir))
        # verify that labels file matches subdirs
        labels_set = frozenset(labels)
        symmetric_difference = subdirs ^ labels_set
        if len(symmetric_difference) != 0:
            print('labels file does not correspond to image subdirectories:')
            print(', '.join(symmetric_difference))
            exit()

        # idx file starts with a magic number
        # 0x8 for unsigned byte, 0x4 for 4 dimensions [images, rows, columns, channels]
        image_magic_number = 0x00000804
        idx_image_file.write(buffer(int_to_bytes(image_magic_number)))
        label_magic_number = 0x00000801
        idx_label_file.write(buffer(int_to_bytes(label_magic_number)))

        # count the images
        image_count = 0
        for label in labels:
            for image_file in glob.glob(os.path.join(args.image_dir, label, '*')):
                image_count += 1

        # TODO discover dimensions
        IMAGE_HEIGHT = 24
        IMAGE_WIDTH = 24
        IMAGE_CHANNELS = 3

        # write dimensions for image file
        idx_image_file.write(buffer(int_to_bytes(image_count)))
        idx_image_file.write(buffer(int_to_bytes(IMAGE_HEIGHT)))
        idx_image_file.write(buffer(int_to_bytes(IMAGE_WIDTH)))
        idx_image_file.write(buffer(int_to_bytes(IMAGE_CHANNELS)))

        # write dimensions for label file
        idx_label_file.write(buffer(int_to_bytes(image_count)))

        label_index = 0
        for label in labels:
            for image_file in glob.glob(os.path.join(args.image_dir, label, '*')):
                pixel_list = []
                image = Image.open(image_file).convert('RGB')
                for pixel in image.getdata():
                    # if type(pixel) is not tuple:
                    #     print(image_file)
                    #     image.show()
                    #     exit()
                    pixel_list.extend(pixel)

                idx_image_file.write(buffer(bytearray(pixel_list)))
                idx_label_file.write(buffer(int_to_ubyte(label_index)))
            label_index += 1



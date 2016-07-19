#!/usr/bin/env python

from PIL import Image
import numpy as np
import sys
import glob

if len(sys.argv) != 3:
    print("Usage: {0} <input file pattern> <output file>".format(sys.argv[0]))
    exit()

input_file_pattern = sys.argv[1]
output_file = sys.argv[2]
print(input_file_pattern)
flat_image_arrays = []

for input_file in glob.glob(input_file_pattern):
    image = Image.open(input_file)
    image_array = np.array(image)
    flat_image_array = image_array.reshape(-1)
    flat_image_arrays.append(np.c_[flat_image_array[None]])

image_stack = np.vstack(flat_image_arrays)
np.savetxt(output_file, image_stack, delimiter=",", fmt="%d")
#!/usr/bin/env python

# A GUI program that makes manually classifying images for test set generation easy.

import argparse
import Tkinter as tk
from PIL import ImageTk, Image
import glob
from collections import deque
import os

class Label:
    def __init__(self, name, keybinding, directory):
        self.name = name
        self.keybinding = keybinding
        self.directory = directory

parser = argparse.ArgumentParser()
parser.add_argument("label_file", help="file with the list of classification labels as lines of a single character key binding followed by the label name")
parser.add_argument("source_image_dir", help="directory containing images to classify")
parser.add_argument("destination_image_dir", help="directory where classified images will be copied")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

# read labels and key bindings
with open(args.label_file) as f:
    lines = f.readlines()
    labels = []
    for line in lines:
        if len(line) <= 1:
            continue
        split = line.split()
        if len(split) != 2 or len(split[0]) != 1:
            print("malformed line:")
            print(line)
            exit()
        labels.append(Label(split[1], split[0], os.path.join(args.destination_image_dir, split[1])))

# show labels
if args.verbose:
    print "key binding - class name - destination directory"
    for label in labels:
        print("{0} - {1} - {2}".format(label.keybinding, label.name, label.directory))

# create destination dirs for each label
for label in labels:
    if not os.path.exists(label.directory):
        os.makedirs(label.directory)
        if args.verbose:
            print("created directory: {0}".format(label.directory))

file_q = deque(glob.glob(args.source_image_dir + "/*"))

top = tk.Tk()
frame = tk.Frame(top, width=100, height=100)

panel = tk.Label(frame)
panel.pack(side="bottom", fill="both", expand="yes")

currently_shown_file_path = None

def callback(event=None):
    # TODO get rid of this global?
    global currently_shown_file_path

    if event is not None:
        # find the label whose key was pressed
        matched_label = next((l for l in labels if l.keybinding == event.char), None)
        if matched_label is None:
            print('unrecognized keystroke: {0}'.format(event.char))
            return

        # move the file
        target_path = os.path.join(matched_label.directory, os.path.basename(currently_shown_file_path))
        if args.verbose:
            print('moving {0} to {1}'.format(currently_shown_file_path, target_path))
        os.rename(currently_shown_file_path, target_path)

    if len(file_q) == 0:
        if args.verbose:
            print("no more files")
        exit()

    filename = file_q.popleft()
    currently_shown_file_path = filename
    img = ImageTk.PhotoImage(Image.open(filename))
    panel.configure(image=img)
    panel.image = img

top.bind("j", callback)
top.bind("k", callback)
frame.pack()

# kickstart it
callback()

top.mainloop()

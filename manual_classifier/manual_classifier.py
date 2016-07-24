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

class Undo:
    def __init__(self, from_path, to_path):
        self.from_path = from_path
        self.to_path = to_path

parser = argparse.ArgumentParser()
parser.add_argument("label_file", help="file with the list of classification labels as lines of a single character key binding followed by the label name")
parser.add_argument("source_image_dir", help="directory containing images to classify")
parser.add_argument("destination_image_dir", help="directory where classified images will be copied")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

reserved_keys = ['u','1','2','3','4','5','6','7','8','9']

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
        key, name = split
        if name in reserved_keys:
            print("key is reserved:")
            print(line)
            exit()
        labels.append(Label(name, key, os.path.join(args.destination_image_dir, name)))

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

file_q = deque(sorted(glob.glob(args.source_image_dir + "/*")))
undo_stack = deque()

top = tk.Tk()
frame = tk.Frame(top, width=200, height=100)

panel = tk.Label(frame)
panel.pack(side="left", fill="both", expand="yes")

panel_next = tk.Label(frame)
panel_next.pack(side="right", fill="both", expand="yes")

currently_shown_file_path = None
repeat_number = None

def move_file(from_path, to_path):
    if args.verbose:
        print('moving {0} to {1}'.format(from_path, to_path))
    os.rename(from_path, to_path)

def callback(event=None):
    # TODO get rid of this global?
    global currently_shown_file_path
    global repeat_number

    if event is not None:
        if event.char == 'u':
            # undo
            if repeat_number is not None:
                print('repeat not supported for undo')
                repeat_number = None

            if len(undo_stack) == 0:
                print("can't undo right now")
                return

            # put the currently shown file back in the queue
            file_q.appendleft(currently_shown_file_path)

            # revert the move and get ready to show the last file again
            undo_info = undo_stack.pop()
            move_file(undo_info.from_path, undo_info.to_path)
            file_q.appendleft(undo_info.to_path)
        elif event.char.isdigit():
            # repeat
            input_num = int(event.char)
            if repeat_number is None:
                repeat_number = input_num
            else:
                repeat_number *= 10
                repeat_number += input_num
            if args.verbose:
                print('repeat mode: {0}'.format(repeat_number))
            return
        else:
            # find the label whose key was pressed
            matched_label = next((l for l in labels if l.keybinding == event.char), None)

            # tell user key is not recognized
            if matched_label is None:
                print('unrecognized keystroke: {0}'.format(event.char))
                return

            files_to_move = [currently_shown_file_path]
            if repeat_number is not None:
                for i in range(repeat_number-1):
                    if len(file_q) == 0:
                        print('repeat goes past end of queue')
                        break
                    files_to_move.append(file_q.popleft())

            for file in files_to_move:
                # move the file
                target_path = os.path.join(matched_label.directory, os.path.basename(file))
                move_file(file, target_path)
                # store paths to perform undo
                undo_stack.append(Undo(target_path, file))

            # reset the repeat
            repeat_number = None

    if len(file_q) == 0:
        if args.verbose:
            print("no more files")
        exit()

    # pop the next file and show it
    filename = file_q.popleft()
    currently_shown_file_path = filename
    img = ImageTk.PhotoImage(Image.open(filename))
    panel.configure(image=img)
    panel.image = img

    # peek the following file and show it in the next pane
    if len(file_q) > 0:
        next_filename = file_q[0]
        img_next = ImageTk.PhotoImage(Image.open(next_filename))
        panel_next.configure(image=img_next)
        panel_next.image = img_next

# u for undo
top.bind('u', callback)
for keybinding in [l.keybinding for l in labels]:
    top.bind(keybinding, callback)
# numbers for repeat
for i in range(10):
    top.bind(str(i), callback)
frame.pack()

# kickstart it
callback()

top.mainloop()

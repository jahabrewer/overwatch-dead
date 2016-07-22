#!/bin/bash

# params: video2stills <video file> <fps> <output filename prefix>
# usage example: video2stills foo.mp4 1/5 out

ffmpeg -i "$1" -vf fps=$2 $3%04d.png

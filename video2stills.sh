#!/bin/bash

# usage example: video2stills foo.mp4 1/5

ffmpeg -i "$1" -vf fps=$2 out%04d.png
#!/bin/bash

# this is meant to be used with xargs for batch mode
# usage example: still2portraits.sh still0013.png still0013
# with xargs: ls | xargs -I file ../still2portraits.sh file file

CROP_X_START=22
CROP_X_SIZE=24
CROP_Y_START=0
CROP_Y_SIZE=24

CONVERT=$(which mogrify)

for FILE in $@; do
    # OUT_FILENAME=${FILE}_${2}.png
    $CONVERT $FILE -crop ${CROP_X_SIZE}x${CROP_Y_SIZE}+${CROP_X_START}+${CROP_Y_START} +repage
done

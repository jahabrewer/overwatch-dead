#!/bin/bash

# this is meant to be used with xargs for batch mode
# usage example: still2portraits.sh still0013.png still0013_
# with xargs: ls | xargs -I file ../still2portraits.sh file file_

CROP_X_STARTS=(15 53 90 127 164 202 404 442 479 517 554 591)
CROP_X_SIZE=34
CROP_Y_START=18
CROP_Y_SIZE=11

CONVERT=$(which convert)

for NUM in $(seq 0 11); do
    CROP_X_START=${CROP_X_STARTS[$NUM]}
    OUT_FILENAME=${2}_${NUM}.png
    $CONVERT $1 -crop ${CROP_X_SIZE}x${CROP_Y_SIZE}+${CROP_X_START}+${CROP_Y_START} +repage $OUT_FILENAME
done
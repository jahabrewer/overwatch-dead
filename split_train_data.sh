#!/bin/bash

for label in $(ls); do
    pushd $label;
    
    n=$(ls | wc -l);
    testsize=$((n / 10));
    destdir=../../crops_test/$label

    if [ ! -d $destdir ]
    then
        mkdir $destdir
    fi

    shuf -zen$testsize * | xargs -0 mv -t $destdir;
    popd;
done
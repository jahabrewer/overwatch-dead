#!/bin/bash

for label in $(ls); do
    pushd $label;
    n=$(ls | wc -l);
    testsize=$((n / 10));
    shuf -zen$testsize * | xargs -0 mv -t ../../crops_test/$label/;
    popd;
done
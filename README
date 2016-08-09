# Quick Start

This CNN achieves 99.93% accuracy on the test set. The two incorrectly classified images are very understandable mixups between the labels dead and none. With scope limited to heroes, it achieves 100% accuracy on the test set.

The test set has only been run once, using this invocation:

overwatch-dead/assets/idx/heroes$ ../../../train.py -v train_images.idx.gz train_labels.idx.gz test_images.idx.gz test_labels.idx.gz -vs 0 -ns 2401 -ci 3000 -b 100 -l 1e-5 -s /tmp/overwatch-logs -t

The important parameters are:
* -vs 0 means nothing in the validation set. There are no hyperparameters to train, so it gets to use all of the train set to train.
* -ns 2401 trains the model for 2401 steps. This takes on the order of 10 minutes on CPU for me.
* -ci 3000 effectively disables the validation checkpoints.
* -b 100 uses 100 examples per training step.
* -l 1e-5 sets the learning rate to 0.00001. This is very important, as higher rates can cause the model to diverge.
* -s /tmp/overwatch-logs is where the tensorboard logs will go.
* -t is most important of all, as it allows the test set to run on the final model.

# Useful things for others
## Dataset
I manually classified about 24,000 hero "portraits" as one of the 24 heroes, dead, or none (no-pick). That dataset is at assets/dataset_heroes.tar.gz.

## Manual classifier tool
To aid me in classifying those 24,000 images, I wrote a tiny tkinter GUI that shows images and lets you classify them with a single keystroke. It also has repeat functionality which is best combined with an open window showing the contents of the image directory for large lookahead. I was able to classify up to 36 images at once, limited by window size, with three keystrokes (3-6-e would mean the next 36 images are Reinhardt). This tool is in manual_classifier.

## images2idx
To take advantage of the existing Google TensorFlow MNIST tutorial code, I wranged my image dataset into IDX format. I found another tool that promised to do this, but it required lots of external libs and I could never get it to compile. This tool is written in python and only requires pillow. It does, unfortunately, require image dimensions to be hardcoded at this point. This tool is at images2idx.py.

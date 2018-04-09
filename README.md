# dehazing
## Data:
So I've got the path to the images hardcoded in solver/train\_val.prototxt,
that'll need to be changed. The .txt files in data folder have the names of the 
images. One thing to note is that for some reason, there's one batch of images
missing - check doit.py. If you're gonna mess with the ITS, it doesn't matter
as all that will have to be modified anyway.

## Solver:
Basically using AOD framework. Just to try to get something functional. What was
provided on their git repo was just for deployment, not training. My modifications
are to try to train. I figure we should be training against un-hazed images so
you can see that I have two extra input layers for those. They're then used in
the loss layer at the end. This results in enormous loss values, so something
is definitely not right. My suspicion right now is that the images need to be
transposed before input as you'll notice that's a common theme for this stuff. 
Not sure how to do that when feeding images using the .txt file stuff. Then, I
could just be totally off. Anyway...

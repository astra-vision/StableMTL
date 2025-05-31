#!/bin/bash

mkdir FlyingThings3D_release
cd FlyingThings3D_release

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_image_clean.tar.bz2
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_flow.tar.bz2
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_disparity.tar.bz2
wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_disparity_change.tar.bz2




# cd ..
# wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip
# unzip FlyingChairs.zip

# wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz
# tar xvzf ChairsSDHom.tar.gz

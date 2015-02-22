#!/bin/sh

URL_PREFIX="https://github.com/smskelley/canny-opencl/releases/download/v1.0-pre1/"
IMAGES="Great_Tit.jpg hs-2004-07-a-full_jpg.jpg lena.jpg world.jpg"

# Decide where we should put the images.
if [ $1 != "" ]
then
  TARGET=$1
else
  if [ -d "images" ]
  then
    TARGET="images"

  elif [ -d "../images" ]
  then
    TARGET="../images"

  else
    mkdir images
    TARGET="images"
  fi
fi

echo "Target: $TARGET"

for i in $IMAGES
do
  wget $URL_PREFIX$i -P $TARGET
done

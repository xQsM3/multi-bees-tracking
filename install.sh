#!/bin/bash


echo "****** create env*****"
conda create -n multi-bees-tracking python=3.8.5
conda activate multi-bees-tracking
echo "****** install tensorflow*****"
pip install tf-nightly-gpu==2.2.0.dev20200508
echo "****** install pytorch*****"
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
echo "****** install detectron2*****"
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
echo "****** install packages*****"
pip install numpy
pip install opencv-contrib-python
pip install sklearn
pip install scikit-learn==0.22.2
pip install ntpath
pip install pandas
pip install tqdm
pip install sty
pip install matplotlib
pip install scipy
pip install plotly

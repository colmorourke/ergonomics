import streamlit as st
import os
import matplotlib.pyplot as plt
#import matlab
from tf_pose.estimator import TfPoseEstimator

from run import pose_estimator
from run import find_point_hw
from run import file_selector
import matplotlib.pyplot as plt
import numpy as np

from PIL import *
from PIL import Image
import PIL.Image

st.title('ERGonomics')


# get full filename with an image


filename = file_selector('./images','Please upload your Finish photo:')

# read jpg
im = Image.open(filename)

# plot image
st.image(im, use_column_width=True)

# get dimensions of the image
width, height = im.size #height,width = im.size # NOTE: im.size returns width, height

st.write("height, width = " + str(height) + " , " + str(width))

st.write(filename)

pose = pose_estimator(filename, True)



filename = file_selector('./images','Please upload your Catch photo:')

# read jpg
im = Image.open(filename)

# plot image
st.image(im, use_column_width=True)

# get dimensions of the image
height,width = im.size

st.write(filename)

pose = pose_estimator(filename, False)





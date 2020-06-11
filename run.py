import argparse
import logging
import sys
import os
import time

import streamlit as st


from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import math
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

blue_color = (0, 35, 102)
pink_color = (255,20,147)

in_finish = True
in_catch = not in_finish



def file_selector(folder_path = './images', str_message='Select your file'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(str_message, filenames)
    return os.path.join(folder_path, selected_filename)

def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)

def find_point_hw(pose, p, height, width):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)

def catch_lean(is_catch, neck_point_x, ear_point_x):
    '''
        Left limbs --> I assume I can't see right limbs
        my lean check is based on if ear is tilted behing neck
    '''
    if is_catch and ear_point_x - neck_point_x > 0:
        return True
    return False

def finish_lean(is_finish, elbow_point_x, nose_point_x):
    '''
        Left limbs --> I assume I can't see right limbs
        my lean check is based on if nose is behind elbow.
        I can of course change/modify this later. 
        The point now is to get the basic thing working.
    '''
    if is_finish and nose_point_x - elbow_point_x > 0:
        return True
    return False

def finish_highpull(is_finish, wrist_y, shoulder_y):
    '''
        my lean check is based on if nose is behind elbow.
        I can of course change/modify this later. 
        The point now is to get the basic thing working.
    '''
    if is_finish and shoulder_y - wrist_y + 5 > 0:
        return True
    return False


def pose_estimator(str_filename, stroke_mode):

    in_finish = stroke_mode
    in_catch  = not in_finish

    nmp_image = common.read_imgfile(str_filename, None, None)

    if nmp_image is None:
        logger.error('Image can not be read, path=%s' % nmp_image)
        sys.exit(-1)

    resize_factor = "432x368"
    model_type = 'mobilenet_thin' 
    args_resize_out_ratio = 4.0

    w, h = model_wh(resize_factor)
    
    e = TfPoseEstimator(get_graph_path(model_type), target_size=(w, h))

    t = time.time()
    humans = e.inference(nmp_image, resize_to_default=(w > 0 and h > 0), upsample_size=args_resize_out_ratio)
    pose = humans
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (nmp_image, elapsed))

    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # height,width = image.shape[0],image.shape[1]

    image_tf = TfPoseEstimator.draw_humans(nmp_image, humans, imgcopy=False)
    height,width = image_tf.shape[0],image_tf.shape[1]

    #height,width = nmp_image.shape[0], nmp_image.shape[1]


    string_1 = "0: Nose = " + str(find_point_hw(pose, 0, height,width))
    st.write(string_1)


    if finish_lean(in_finish, find_point_hw(pose, 6, height, width)[0], find_point_hw(pose, 0, height, width)[0]):
        action1 = "Leaning back too far at the finish!"
        draw_str(nmp_image, (5, 100), action1, pink_color, 6)
        st.write(action1)

    if finish_highpull(in_finish, find_point_hw(pose, 7, height, width)[1], find_point_hw(pose, 5, height,width)[1]):
        action2 = "You're pulling your hands in too high!"
        draw_str(nmp_image, (5, 180), action2, pink_color, 6)
        st.write(action2)

    if catch_lean(in_catch, find_point_hw(pose, 1, height,width)[0], find_point_hw(pose, 17, height, width)[0]):
        action = "You need to lean forward slightly at the catch!"
        draw_str(nmp_image, (5, 180), action, pink_color, 5)


    fig = plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(nmp_image, cv2.COLOR_BGR2RGB))
    plt.subplots_adjust(left=0, right=0.1, top=0.9, bottom=0.1)
    fig.tight_layout()
    st.pyplot()
    #st.pyplot(use_column_width=True)


    # except Exception as e:
    #     logger.warning('matplotlib error, %s' % e)
    #     cv2.imshow('result', nmp_image)
    #     cv2.waitKey()

    return pose 
    # return find_point(pose, 0)

    


def draw_str(dst, xxx_todo_changeme, s, color, scale):
    
    (x, y) = xxx_todo_changeme
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    #cv2.line    
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='tf-pose-estimation run')
#     parser.add_argument('--image', type=str, default='./images/p1.jpg')
#     parser.add_argument('--model', type=str, default='cmu',
#                         help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
#     parser.add_argument('--resize', type=str, default='0x0',
#                         help='if provided, resize images before they are processed. '
#                              'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
#     parser.add_argument('--resize-out-ratio', type=float, default=4.0,
#                         help='if provided, resize heatmaps before they are post-processed. default=1.0')

#     args = parser.parse_args()

#     w, h = model_wh(args.resize)
#     if w == 0 or h == 0:
#         e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
#     else:
#         e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

#     print(args.image)

#     # estimate human poses from a single image !
#     image = common.read_imgfile(args.image, None, None)
#     print(type(image))
#     if image is None:
#         logger.error('Image can not be read, path=%s' % args.image)
#         sys.exit(-1)

#     t = time.time()
#     humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
#     pose = humans
#     elapsed = time.time() - t


#     for point in pose:
#         print(type(point))
#         print(point.body_parts[0])

#     logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

#     image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
#     height,width = image.shape[0],image.shape[1]

#     print("height, width = " + str(height) + ", " +  str(width))
#     print("0: Nose = " + str(find_point(pose, 0)))
#     print("15: Eye Left = " + str(find_point(pose, 15)))
#     print("17: Ear Left = " + str(find_point(pose, 17)))
#     print("1: Neck = " + str(find_point(pose, 1)))
#     print("5: Shoulder Left = " + str(find_point(pose, 5)))
#     print("6: Elbow Left = " + str(find_point(pose, 6)))
#     print("7: Wrist Left = " + str(find_point(pose, 7)))
#     print("8: Foot Left = " + str(find_point(pose, 13)))

#     try:
        

#         if finish_lean(in_finish, find_point(pose, 6)[0], find_point(pose, 0)[0]):
#            action = "Leaning back too far at the finish!"
#            draw_str(image, (5, 100), action, pink_color, 6)

#         if finish_highpull(in_finish, find_point(pose, 7)[1], find_point(pose, 5)[1]):
#            action = "You're pulling your hands in too high!"
#            draw_str(image, (5, 180), action, pink_color, 6)

#         if catch_lean(in_catch, find_point(pose, 1)[0], find_point(pose, 17)[0]):
#            action = "You need to lean forward slightly at the catch!"
#            draw_str(image, (5, 180), action, pink_color, 5)

#         fig = plt.figure()
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.show()



#     except Exception as e:
#         logger.warning('matplotlib error, %s' % e)
#         cv2.imshow('result', image)
#         cv2.waitKey()

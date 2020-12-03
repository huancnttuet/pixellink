import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
slim = tf.contrib.slim


def basenet(inputs, fatness = 64, dilation = True):
    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        # Block1
        print(inputs)
        net = slim.repeat(inputs, 2, slim.conv2d, fatness, [3, 3], scope='conv1')
        print(net)
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['pool1'] = net
        block1 = net
        
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, fatness * 2, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        print(net)        
        block2 = net
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 4, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        print(net)        
        block3 = net        
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv4')
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        end_points['pool4'] = net
        print(net)        
        block4 = net
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        end_points['pool5'] = net
        print(net)        
        block5 = net
        # fc6 as conv, dilation is added
        if dilation:
            net = slim.conv2d(net, fatness * 16, [3, 3], rate=6, scope='fc6')
        else:
            net = slim.conv2d(net, fatness * 16, [3, 3], scope='fc6')
        end_points['fc6'] = net
        print(net)        
        fc6 = net
        # fc7 as conv
        net = slim.conv2d(net, fatness * 16, [1, 1], scope='fc7')
        end_points['fc7'] = net
        print(net)        
        fc7 = net
    return net, end_points, block1, block2, block3, block4, block5, fc6, fc7;    


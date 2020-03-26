from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
import numpy as np


# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets, image_count, steps_per_epoch')

        #Check the input directory
        if (FLAGS.input_dir_LR == 'None') or (FLAGS.input_dir_HR == 'None'):
            raise ValueError('Input directory is not provided')

        if (not os.path.exists(FLAGS.input_dir_LR)) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')

        image_list_LR = os.listdir(FLAGS.input_dir_LR)
        image_list_LR = [_ for _ in image_list_LR if _.endswith('.png')]
        if len(image_list_LR)==0:
            raise Exception('No png files in the input directory')

        image_list_LR_temp = sorted(image_list_LR)
        image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp]
        image_list_HR = [os.path.join(FLAGS.input_dir_HR, _) for _ in image_list_LR_temp]

        image_list_LR_tensor = tf.convert_to_tensor(image_list_LR, dtype=tf.string)
        image_list_HR_tensor = tf.convert_to_tensor(image_list_HR, dtype=tf.string)

        with tf.variable_scope('load_image'):
            output = tf.train.slice_input_producer([image_list_LR_tensor, image_list_HR_tensor],
                                                   shuffle=False, capacity=FLAGS.name_queue_capacity)


            # Reading and decode the HR images
            reader = tf.WholeFileReader(name='image_reader')
            image_LR = tf.read_file(output[0])
            image_HR = tf.read_file(output[1])
            input_image_LR = tf.image.decode_png(image_LR, channels=1)
            input_image_HR = tf.image.decode_png(image_HR, channels=1)
            input_image_LR = tf.image.convert_image_dtype(input_image_LR, dtype=tf.float32)
            input_image_HR = tf.image.convert_image_dtype(input_image_HR, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(input_image_LR)[2], 1, message="image should only have 1 channel")
            with tf.control_dependencies([assertion]):
                if FLAGS.interpolation == True:
                    input_image_LR = tf.image.resize_images(input_image_LR,[FLAGS.crop_size*2,FLAGS.crop_size*2])
                else:
                    input_image_LR = tf.identity(input_image_LR)
                input_image_HR = tf.identity(input_image_HR)

            # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
            a_image = preprocessLR(input_image_LR)
            b_image = preprocess(input_image_HR)

            inputs, targets = [a_image, b_image]


        with tf.name_scope('data_preprocessing'):
            input_images = tf.identity(input_image_LR)
            target_images = tf.identity(input_image_HR)


            if FLAGS.interpolation == True:
                input_images.set_shape([FLAGS.crop_size*2, FLAGS.crop_size*2, 1])
            else:
                input_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 1])
            target_images.set_shape([FLAGS.crop_size*2, FLAGS.crop_size*2, 1])

        if FLAGS.mode == 'train':
            paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.shuffle_batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, capacity=FLAGS.image_queue_capacity+4*FLAGS.batch_size,
                                            min_after_dequeue=FLAGS.image_queue_capacity, num_threads=FLAGS.queue_thread)
        else:
            paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, num_threads=FLAGS.queue_thread, allow_smaller_final_batch=True)

        steps_per_epoch = int(math.ceil(len(image_list_LR) / FLAGS.batch_size))
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'unet':
            if FLAGS.interpolation == True:
                inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size*2, FLAGS.crop_size*2, 1])
            else:
                inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 1])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size*2, FLAGS.crop_size*2, 1])

    return Data(
        paths_LR=paths_LR_batch,
        paths_HR=paths_HR_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=len(image_list_LR),
        steps_per_epoch=steps_per_epoch
    )


# The test data loader. Allow input image with different size
def test_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_LR == 'None') or (FLAGS.input_dir_HR == 'None'):
        raise ValueError('Input directory is not provided')

    if (not os.path.exists(FLAGS.input_dir_LR)) or (not os.path.exists(FLAGS.input_dir_HR)):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(FLAGS.input_dir_LR)
    image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']
    image_list_HR = [os.path.join(FLAGS.input_dir_HR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    # Read in and preprocess the images
#    def preprocess_test(name, mode):
#        im = sic.imread(name, mode="RGB").astype(np.float32)
#        # check grayscale image
#        if im.shape[-1] != 1:
#            h, w = im.shape
#            temp = np.empty((h, w, 1), dtype=np.uint8)
#            temp[:, :, :] = im[:, :, np.newaxis]
#            im = temp.copy()
#        if mode == 'LR':
#            im = im / np.max(im)
#        elif mode == 'HR':
#            im = im / np.max(im)
#            im = im * 2 - 1
#
#        return im
# Read in and preprocess the images
def preprocess_test(name, mode):
    im = sic.imread(name, mode="L").astype(np.float32)/255
    im = im[:,:,np.newaxis]
    
    return im

    image_LR = [preprocess_test(_, 'LR') for _ in image_list_LR]
    image_HR = [preprocess_test(_, 'HR') for _ in image_list_HR]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets')

    return Data(
        paths_LR = image_list_LR,
        paths_HR = image_list_HR,
        inputs = image_LR,
        targets = image_HR
    )


# The inference data loader. Allow input image with different size
def inference_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_LR == 'None'):
        raise ValueError('Input directory is not provided')

    if not os.path.exists(FLAGS.input_dir_LR):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(FLAGS.input_dir_LR)
    image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    # Read in and preprocess the images
    def preprocess_test(name):
        im = sic.imread(name, mode="L").astype(np.float32)/255
        print('im.shape=',im.shape)
        im = im[:,:,np.newaxis]
        #im = tf.image.resize_images(im,[FLAGS.crop_size*2,FLAGS.crop_size*2])

        # check grayscale image
        
#        if im.shape[-1] != 1:
#            h, w = im.shape
#            temp = np.empty((h, w, 1), dtype=np.uint8)
#            temp[:, :, :] = im[:, :, np.newaxis]
#            im = temp.copy()
#        h, w = im.shape
#        im = sic.imresize(im,(2*h,2*w),'cubic')
#        im = im / np.max(im)


        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]
    print('image_LR=',image_LR)
    # list to array to tensor
    image_LR = np.asarray(image_LR)
    print('image_LR(array) = ',image_LR.shape)
    image_LR = tf.image.resize_images(image_LR,[256*2,256*2])
    # tensor to array to list
    print('image_LR(tensor).shape=', image_LR.shape)
    with tf.Session() as sess:
        image_LR = image_LR.eval()
    image_LR = np.ndarray.tolist(image_LR)

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


# Definition of the generator
import lib.ops as ops
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # define u-net
    def unet(input, output_channel, scope):
        with slim.arg_scope([slim.conv2d,  slim.conv2d_transpose, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            # slim.conv2d default relu activation
            # subsampling
            convl0 = slim.repeat(gen_inputs, 2, slim.conv2d, 64, [3, 3], scope='convl0')
            pool0 = slim.max_pool2d(convl0, [2, 2], scope='pool0')
            bn0 = slim.batch_norm(pool0, decay=0.9, epsilon=1e-5, scope="bn0")

            convl1 = slim.repeat(bn0, 2, slim.conv2d, 128, [3, 3], scope='convl1')
            pool1 = slim.max_pool2d(convl1, [2, 2], scope='pool1')
            bn1 = slim.batch_norm(pool1, decay=0.9, epsilon=1e-5, scope="bn1")

            # upsampling
            conv_t1 = slim.conv2d_transpose(bn1, 256, [2,2], scope='conv_t1')
            merge1 = tf.concat([conv_t1, convl1], 3)
            convl2 = slim.stack(merge1, slim.conv2d, [(256, [3, 3]),(128, [3,3])], scope='convl2')
            bn2 = slim.batch_norm(convl2, decay=0.9, epsilon=1e-5, scope='bn2')

            conv_t2 = slim.conv2d_transpose(bn2, 128, [2,2], scope='conv_t2')
            merge2 = tf.concat([conv_t2, convl0], 3)
            convl3 = slim.stack(merge2, slim.conv2d, [(128, [3,3]), (64, [3,3])], scope='convl3')
            bn3 = slim.batch_norm(convl3, decay=0.9, epsilon=1e-5, scope='bn3')

            # output layer scoreMap
            net = slim.conv2d(bn3, 1, [1,1], scope='scoreMap')
        return net


    # generator
    with tf.variable_scope('generator_unit', reuse=reuse):
        with tf.variable_scope('unet_output'):
            net = unet(gen_inputs, gen_output_channels, 'unet')

        #with tf.variable_scope('subpixelconv_stage1'):
        #    net = ops.convol2(net, 3, 64, 1, scope='conv')
        #    net = ops.pixelShuffler(net, scale=2)
        #    net = ops.prelu_tf(net)

        #with tf.variable_scope('output_stage'):
        #    net = ops.convol2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = convol2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = convol2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net



# Define the whole network architecture
def SRGAN(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, \
        discrim_grads_and_vars, adversarial_loss, content_loss, gen_grads_and_vars, gen_output, train, global_step, \
        learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size*2, FLAGS.crop_size*2, 1])

    # Build the fake discriminator
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)

    # Build the real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            discrim_real_output = discriminator(targets, FLAGS=FLAGS)


    # Use MSE loss directly
    if FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type!!')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.vgg_scaling*tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + FLAGS.EPS))

        gen_loss = content_loss + (FLAGS.ratio)*adversarial_loss
        print(adversarial_loss.get_shape())
        print(content_loss.get_shape())

    # Calculating the discriminator loss
    with tf.variable_scope('discriminator_loss'):
        discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
        discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)

        discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train'):
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
        discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies([discrim_train]+ tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    #[ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        discrim_real_output = discrim_real_output,
        discrim_fake_output = discrim_fake_output,
        discrim_loss = exp_averager.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        adversarial_loss = exp_averager.average(adversarial_loss),
        content_loss = exp_averager.average(content_loss),
        gen_grads_and_vars = gen_grads_and_vars,
        gen_output = gen_output,
        train = tf.group(update_loss, incr_global_step, gen_train),
        global_step = global_step,
        learning_rate = learning_rate
    )


def unet(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'content_loss, gen_grads_and_vars, gen_output, train, global_step, \
            learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size * 2, FLAGS.crop_size * 2, 1])

    # Use the VGG54 feature
    if FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        gen_loss = content_loss

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # [ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([content_loss])

    return Network(
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate
    )


def save_images(fetches, FLAGS, step=None):
    image_dir = os.path.join(FLAGS.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

    if FLAGS.mode == 'inference':
        kind = "outputs"
        filename = name + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

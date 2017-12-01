import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys


class DTN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.mode = mode
        self.learning_rate = learning_rate
        
    def content_extractor_old(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train' or self.mode=='pretrain')):
                    
                    net = slim.conv2d(images, 64, [3, 3], scope='conv1')   # (batch_size, 16, 16, 64)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')     # (batch_size, 8, 8, 128)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')     # (batch_size, 4, 4, 256)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 128, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 128)
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')
                    if self.mode == 'pretrain':
                        net = slim.conv2d(net, 10, [1, 1], padding='VALID', scope='out')
                        net = slim.flatten(net)
                    return net


    def content_extractor(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train' or self.mode=='pretrain')):
                    
                    net = slim.conv2d(images, 32, [3, 3], scope='conv1')   # (batch_size, 16, 16, 32)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 16, [3, 3], scope='conv2')     # (batch_size, 8, 8, 16)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 8, [3, 3], scope='conv3')     # (batch_size, 4, 4, 8)
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn3')
                    # print "***" + str(net.shape)
                    # net = slim.conv2d(net, 128, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 128)
                    # net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')

                    if self.mode == 'pretrain':
                        ### Decoder
                        upsample1 = tf.image.resize_images(net, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        # print "---" + str(upsample1.shape)
                        # Now 8x8x32
                        conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                        # print "---" + str(conv4.shape)
                        # Now 8x8x16
                        upsample2 = tf.image.resize_images(conv4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        # print "---" + str(upsample2.shape)
                        # Now 16x16x16
                        conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                        # print "---" + str(conv5.shape)
                        # Now 16x16x32
                        upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        # print "---" + str(upsample3.shape)
                        # Now 32x32x32
                        conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                        # print "---" + str(conv6.shape)
                        # Now 32x32x32

                        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
                        # print "---" + str(logits.shape)
                        # Now 32x32x3
                        net = logits
                    else:
                        net = tf.reshape(net, [-1,1,1,128])
                return net

    def content_extractor_1(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                #encoder
                net = tf.layers.conv2d(images, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # print "**1" + str(net.shape)
                 # (batch_size, 16, 16, 32)
                net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(2,2), padding='same')
                # print "**2" + str(net.shape)
                # (batch_size, 8, 8, 32)
                net = tf.layers.conv2d(net, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # print "**3" + str(net.shape)
                # (batch_size, 4, 4, 32)
                net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(2,2), padding='same')
                # print "**4" + str(net.shape)
                # (batch_size, 2, 2, 32)
                net = tf.layers.conv2d(net, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # print "**5" + str(net.shape)
                # (batch_size, 1, 1, 8)
                net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(2,2), padding='same')
                # print "**6" + str(net.shape)
                # (batch_size, 1, 1, 8)

                if self.mode == 'pretrain':
                    ### Decoder
                    upsample1 = tf.image.resize_images(net, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    # print "---" + str(upsample1.shape)
                    # Now 8x8x32
                    conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                    # print "---" + str(conv4.shape)
                    # Now 8x8x16
                    upsample2 = tf.image.resize_images(conv4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    # print "---" + str(upsample2.shape)
                    # Now 16x16x16
                    conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                    # print "---" + str(conv5.shape)
                    # Now 16x16x32
                    upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    # print "---" + str(upsample3.shape)
                    # Now 32x32x32
                    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                    # print "---" + str(conv6.shape)
                    # Now 32x32x32

                    logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
                    # print "---" + str(logits.shape)
                    # Now 32x32x3
                    net = logits
                else:
                    net = tf.reshape(net, [-1,1,1,128])
                return net



                # ### Encoder
                # conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 28x28x32
                # maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
                # # Now 14x14x32
                # conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 14x14x32
                # maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
                # # Now 7x7x32
                # conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 7x7x16
                # encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
                # # Now 4x4x16

                # ### Decoder
                # upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # # Now 7x7x16
                # conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 7x7x16
                # upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # # Now 14x14x16
                # conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 14x14x32
                # upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # # Now 28x28x32
                # conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
                # # Now 28x28x32


                # logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
                # #Now 28x28x1
                # # Pass logits through sigmoid to get reconstructed image
                # decoded = tf.nn.sigmoid(logits)
                # # Pass logits through sigmoid and calculate the cross-entropy loss
                # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
                # # Get cost and define the optimizer
                # cost = tf.reduce_mean(loss)
                # opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
                
    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128)
        # print "***" + str(inputs.shape)
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 1, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    return net
    
    def discriminator(self, images, reuse=False):
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(images, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv1')   # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')   # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv3')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 1)
                    net = slim.flatten(net)
                    return net
                
    def build_model(self):
        
        if self.mode == 'pretrain_old':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            # Adding MNIST images for pretraining f
            mnist_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            mnist_images = tf.image.grayscale_to_rgb(mnist_images)
            self.images += mnist_images
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)

            self.loss = tf.reduce_mean(tf.square(self.images - self.logits))
            # pred=tf.reshape(self.logits,[-1,32*32*3])
            # y=tf.reshape(self.images,[-1,32*32*3])
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))

            self.accuracy = self.loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            
            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.src_images)
            self.fake_images = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_images)
            self.fgfx = self.content_extractor(self.fake_images, reuse=True)

            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0
            
            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]
            
            # train op
            with tf.variable_scope('source_train_op',reuse=False):
                self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=f_vars)
            
            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
                                                    f_loss_src_summary, origin_images_summary, 
                                                    sampled_images_summary])
            
            # target domain (mnist)
            self.fx = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits_real = self.discriminator(self.trg_images, reuse=True)
            
            # loss
            self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * 15.0
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg
            
            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.variable_scope('target_train_op',reuse=False):
                self.d_train_op_trg = slim.learning.create_train_op(self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)
            
            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary, 
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_images_summary, sampled_images_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            
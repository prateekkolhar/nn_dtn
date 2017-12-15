import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from random import randint


class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=10000, train_iter=2000, sample_iter=100,
                 svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', pretrain_sample_save_path= 'pretrain_sample',pretrained_model='model/svhn_model-10000', test_model='model/dtn-200'):

        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.pretrain_sample_save_path = pretrain_sample_save_path

    def load_svhn(self, image_dir, split='train'):
        print ('loading svhn image dataset..')

        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'

        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        print ('finished loading svhn image dataset..!')
        return images, labels

    def load_mnist(self, image_dir, split='train'):
        print ('loading mnist image dataset..')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print ('finished loading mnist image dataset..!')
        return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain_old(self):
        # load svhn dataset
        train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict)

                if (step+1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                                           feed_dict={model.images: test_images[rand_idxs],
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1)
                    print ('svhn_model-%d saved..!' %(step+1))

    def pretrain(self):
        # load svhn dataset
        train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op1, feed_dict)
                sess.run(model.train_op2, feed_dict)

                if (step+1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss2, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss2],
                                           feed_dict={model.images: test_images[rand_idxs],
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1)
                    print ('svhn_model-%d saved..!' %(step+1))
    # def pretrain_plot(self,img, file_name='model/svhn_model-1000'):
    # 	print ('loading pretrained model F..')
    #     variables_to_restore = slim.get_model_variables(scope='content_extractor')
    #     restorer = tf.train.Saver(variables_to_restore)
    #     restorer.restore(sess, file_name)




    def train(self):
        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir, split='train')
        mnist_images, _ = self.load_mnist(self.mnist_dir, split='train')

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=0)

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):

                i = step % int(svhn_images.shape[0] / self.batch_size)
                # train the model for source domain S
                src_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.src_images: src_images}

                sess.run(model.d_train_op_src, feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)

                if step > 1600:
                    f_interval = 30

                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))

                # train the model for target domain T
                j = step % int(mnist_images.shape[0] / self.batch_size)
                trg_images = mnist_images[j*self.batch_size:(j+1)*self.batch_size]
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl))

                if (step+1) % 100 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1 )
                    print ('model/dtn-%d saved' %(step+1))

    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)

    def pretrain_eval_s(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir)


        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.pretrained_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.pretrain_sample_save_path, 's_sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)

    def pretrain_eval_t(self):
        # build model
        model = self.model
        model.build_model()

        # load mnist dataset
        mnist_images, _ = self.load_mnist(self.mnist_dir)


        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.pretrained_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = mnist_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.pretrain_sample_save_path, 't_sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)

    def get_random_image(self, images, labels, category):
        filtered_images = self.get_all_images(images, labels, category)
        return filtered_images[randint(0, len(filtered_images) - 1)]

    def get_all_images(self, images, labels, category):
        filtered_images = []
        for i, label in enumerate(labels):
            if label == category:
                filtered_images.append(images[i])
        return filtered_images

    def pretrain_eval_separation(self):
        # load svhn dataset
        svhn_images, svhn_labels = self.load_svhn(self.svhn_dir)
        mnist_images, mnist_labels = self.load_mnist(self.mnist_dir)

        # build a graph
        model = self.model
        model.build_model()

        global_loss_vec = []
        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.pretrained_model)

            for i in xrange(10):
                # m_image = self.get_random_image(mnist_images, mnist_labels, i)
                # m_images = [m_image]
                m_images = self.get_all_images(mnist_images, mnist_labels, i)
                for j in xrange(10):
                    # s_image = self.get_random_image(svhn_images, svhn_labels, j)
                    # s_images = [s_image]
                    s_images = self.get_all_images(svhn_images, svhn_labels, j)
                    fs, ft = sess.run(fetches=[model.fs, model.ft],
                                    feed_dict={model.src_images: s_images,
                                               model.trg_images: m_images})
                    mean_fs = np.mean(fs, 0)
                    mean_ft = np.mean(ft, 0)
                    loss = np.mean(np.square(mean_fs - mean_ft))
                    var_s = np.var(fs, 0)
                    var_t = np.var(ft, 0)
                    print "loss for " + str(i) + " vs " + str(j) + " is " + str(loss)
                    print "var_s: " + str(var_s) + " and var_t: " + str(var_t)

    def pretrain_eval_separation_after_test(self):
        # load svhn dataset
        svhn_images, svhn_labels = self.load_svhn(self.svhn_dir)
        mnist_images, mnist_labels = self.load_mnist(self.mnist_dir)

        # build a graph
        model = self.model
        model.build_model()

        global_loss_vec = []
        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for i in xrange(10):
                # m_image = self.get_random_image(mnist_images, mnist_labels, i)
                # m_images = [m_image]
                m_images = self.get_all_images(mnist_images, mnist_labels, i)
                for j in xrange(10):
                    # s_image = self.get_random_image(svhn_images, svhn_labels, j)
                    # s_images = [s_image]
                    s_images = self.get_all_images(svhn_images, svhn_labels, j)
                    fs, ft = sess.run(fetches=[model.fs, model.ft],
                                    feed_dict={model.src_images: s_images,
                                               model.trg_images: m_images})
                    mean_fs = np.mean(fs, 0)
                    mean_ft = np.mean(ft, 0)
                    loss = np.mean(np.square(mean_fs - mean_ft))
                    var_s = np.var(fs, 0)
                    var_t = np.var(ft, 0)
                    print "loss for " + str(i) + " vs " + str(j) + " is " + str(loss)
                    print "var_s: " + str(var_s) + " and var_t: " + str(var_t)

    def pretrain_intra_variance(self):
        # load svhn dataset
        svhn_images, svhn_labels = self.load_svhn(self.svhn_dir)
        mnist_images, mnist_labels = self.load_mnist(self.mnist_dir)

        # build a graph
        model = self.model
        model.build_model()

        global_loss_vec = []
        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.pretrained_model)

            for i in xrange(10):
                # m_image = self.get_random_image(mnist_images, mnist_labels, i)
                # m_images = [m_image]
                m_images = self.get_all_images(mnist_images, mnist_labels, i)
                s_images = self.get_all_images(svhn_images, svhn_labels, i)
                fs, ft = sess.run(fetches=[model.fs, model.ft],
                                  feed_dict={model.trg_images: m_images, model.src_images: s_images})

                var_fs = np.var(fs, 0)
                var_ft = np.var(ft, 0)
                print str(np.mean(var_fs)) + "\t" + str(np.mean(var_ft))

    def pretrain_intra_variance_after_test(self):
        # load svhn dataset
        svhn_images, svhn_labels = self.load_svhn(self.svhn_dir)
        mnist_images, mnist_labels = self.load_mnist(self.mnist_dir)

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for i in xrange(10):
                # m_image = self.get_random_image(mnist_images, mnist_labels, i)
                # m_images = [m_image]
                m_images = self.get_all_images(mnist_images, mnist_labels, i)
                s_images = self.get_all_images(svhn_images, svhn_labels, i)
                fs, ft = sess.run(fetches=[model.fs, model.ft],
                                  feed_dict={model.trg_images: m_images, model.src_images: s_images})

                var_fs = np.var(fs, 0)
                var_ft = np.var(ft, 0)
                print str(np.mean(var_fs)) + "\t" + str(np.mean(var_ft))
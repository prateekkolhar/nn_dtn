import tensorflow as tf
from model import DTN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'pretrain_eval_s', 'pretrain_eval_t', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('pretrain_sample_save_path', 'pretrain_sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS

def main(_):
    
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003)
    solver = Solver(model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                    svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path,
                   pretrain_sample_save_path=FLAGS.pretrain_sample_save_path)
    
    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)
    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'pretrain_eval_s':
        solver.pretrain_eval_s()
    elif FLAGS.mode == 'pretrain_eval_t':
        solver.pretrain_eval_t()
    elif FLAGS.mode == 'train':
        solver.train()
    elif FLAGS.mode == "pretrain_eval_separation":
        solver.pretrain_eval_separation()
    else:
        solver.eval()
        
if __name__ == '__main__':
    tf.app.run()
#!/usr/bin/python3
import tensorflow as tf     #import general python packages
import numpy as np
import pandas as pd
import time, os, sys
import argparse


from network import Network     #import components from other user defined python files
from utils import Utils
from data import Data
from model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)      #sets the type of logs to be displayed. "error" limits logging messages to occurence of error. other modes are "debug", "info" "warn" & "fatal"

def train(config, args):
    print('Inside train fn of train.py')  
    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf') # https://stackoverflow.com/questions/34264710/what-is-the-point-of-floatinf-in-python
    
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints) # checks for checkpoint in the path 'directories.checkpoints'( from the directories object in config.py) 
                                                                  #https://www.tensorflow.org/api_docs/python/tf/train/get_checkpoint_state
    print('check point state is : ',ckpt) #https://web.stanford.edu/class/cs20si/2017/lectures/notes_05.pdf
    # Load data
    print('Training on dataset', args.dataset) #args.dataset is by default set to cityscapes, as given in main() function.
    if config.use_conditional_GAN:  #by default set to false in config.py. change to true if needed.
        print('Using conditional GAN')
        paths, semantic_map_paths = Data.load_dataframe(directories.train, load_semantic_maps=True)
        test_paths, test_semantic_map_paths = Data.load_dataframe(directories.test, load_semantic_maps=True)
    else:
        paths = Data.load_dataframe(directories.train)
        test_paths = Data.load_dataframe(directories.test)
        #paths and test_paths are numpy.ndarray.

    # Build graph
    gan = Model(config, paths, name=args.name, dataset=args.dataset) 
    '''
    gan is an object of  Model class.
    config is the object for hyperparameters.
    paths is the numpy array for train images.
    name is passed from args, and is set to gan-train by default.
    dataset is passed from args, and is set to cityscapes by default. 
    '''
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths, 
                               gan.test_semantic_map_path_placeholder: test_semantic_map_paths}
        feed_dict_train_init = {gan.path_placeholder: paths,
                                gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths}
        feed_dict_train_init = {gan.path_placeholder: paths}

##########################################################################################

    print('/////////////// Just before creating session ///////////////')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("////////////after variables are initialized")
        train_handle = sess.run(gan.train_iterator.string_handle())
        test_handle = sess.run(gan.test_iterator.string_handle())
        print("////////////after handles are initialized")
        #sess.run(gan.ab)
        #print('9999999999999999999999 ab eval 999999999999999999999999',gan.ab.eval())
        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))
        print("////////////after checkpoints are initialized")        
        sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)
        i=0
        for epoch in range(config.num_epochs):
            print("/////////////////inside epoch loop")
            i=i+1
            print("iteration no ",i)
            sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)

            # Run diagnostics
            G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle,
                start_time, epoch, args.name, G_loss_best, D_loss_best)

            while True:
                try:
                    # Update generator
                    # for _ in range(8):
                    feed_dict = {gan.training_phase: True, gan.handle: train_handle}
                    sess.run(gan.G_train_op, feed_dict=feed_dict)

                    # Update discriminator 
                    step, _ = sess.run([gan.D_global_step, gan.D_train_op], feed_dict=feed_dict)
                    
                    if step % config.diagnostic_steps == 0:
                        G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle,
                            start_time, epoch, args.name, G_loss_best, D_loss_best)
                        Utils.single_plot(epoch, step, sess, gan, train_handle, args.name, config)
                        # for _ in range(4):
                        #    sess.run(gan.G_train_op, feed_dict=feed_dict)


                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        '{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.joinamen(directories.checkpoints,
                               '{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):   # * and ** are used to pass non named and named arguments respectively, of a variable length. note that *(name_of_variable) works as a list and **(name_of_variable) as a dict
    parser = argparse.ArgumentParser()  #https://docs.python.org/3/howto/argparse.html - for info about next few lines
    print("type of parser",type(parser))
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    args = parser.parse_args()

    # Launch training
    train(config_train, args) #config_train is a class in config.py which contains hyperparameters. args is an argparse.ArgumentParser() object.

if __name__ == '__main__':    # at the start of execution, python has some global variables defined. if it is being used as a program, __name__ is set to __main__, otherwise if used as module, its
    main()                    # assigned the name of the file.


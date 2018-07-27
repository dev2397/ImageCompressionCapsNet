
import tensorflow as tf
from utils import Utils
from capsLayer import CapsLayer

def capsule_discriminator(x, config, training, actv=tf.nn.leaky_relu, use_sigmoid=False, ksize=4, mode='real', reuse=False):            

    if mode == 'real':
        print('Mode real Building discriminator D(x)')
    elif mode == 'reconstructed':
        print('Mode reconstructed Building discriminator D(G(z))')
    else:
        raise NotImplementedError('Invalid discriminator mode specified.')    
    
    
    x2 = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')
    x4 = tf.layers.average_pooling2d(x2, pool_size=3, strides=2, padding='same')
    
    def discriminator(x, scope, actv=actv, use_sigmoid=use_sigmoid, ksize=ksize, reuse=reuse):
        
        with tf.variable_scope('discriminator_{}'.format(scope), reuse=reuse):       
            output = tf.reshape(inputs, [-1, 1, 512, 1024])
            output = tf.transpose(output, [0, 2, 3, 1])
            
            output_conv1 = tf.contrib.layers.conv2d(output, num_outputs=256, kernel_size=9, stride=2, activation_fn=tf.nn.relu, padding='VALID') 
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV') 
            output_caps1 = primaryCaps(output_conv1, kernel_size=9, stride=2, batchsize=batchsize)  
            digitCaps = CapsLayer(num_outputs=1, vec_len=16, with_routing=True, layer_type='FC')
            output_caps2 = digitCaps(output_caps1, batchsize=batchsize)

            # The output at this stage is of dimensions [batch_size, 16]
            output_caps2 = tf.squeeze(output_caps2, axis=1)
            output_caps2 = tf.squeeze(output_caps2, axis=2)
       
            #print(output_caps2.get_shape())
            assert output_caps2.get_shape() == [batchsize, 16] # [batchsize,16] turns into 

            # TODO: Try also removing the LeakyReLU from the CapsLayer file
            # TODO: Try also with 10 digitcaps outputs + thresholding (instead of just 1 output)
            # TODO: Adding batch normalization in capsules (See CapsLayer.py). 
            # TODO: Try Changing the critic iteration count.

            output_v_length = tf.sqrt(tf.reduce_sum(tf.square(output_caps2),axis=1, keep_dims=True) + 1e-9)

            ## No need to take softmax anymore, because output_caps2 output is in [0,1] due to squash function.   
            #softmax_v = tf.nn.softmax(v_length, dim=1)

            return tf.reshape(output_v_length, [-1])
    
    with tf.variable_scope('discriminator', reuse=reuse):
        print('Inside ajh * * * * * ')
        disc'''*Dk''' = discriminator(x, 'original')
        disc_downsampled_2''' *Dk_2''' = discriminator(x2, 'downsampled_2')
        disc_downsampled_4''' *Dk_4''' = discriminator(x4, 'downsampled_4')

    return disc, disc_downsampled_2, disc_downsampled_4''' Dk, Dk_2, Dk_4'''
            
            



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         

import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling, AvgPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *
from nonlocal_resnet_utils import nonlocal_dot
from .encoders import inception_encoder, densenet_encoder

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'

    
############## IF USING NON-LOCAL ####################
# Change to channels_last format
# Change data format to 'NHWC'
# Comment out: 
#i = tf.transpose(images, [0, 3, 1, 2]) and replace it with: i = images
# Comment out: d[0] = crop_op(d[0], (184, 184)) d[1] = crop_op(d[1], (72, 72))
#Padding = same in decoder
#######################################################

    
    
####
def upsample2x(name, x):
    """
    Nearest neighbor up-sampling
    """
    return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                data_format='channels_first')
####


def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
##########################################
#Resnet50 block
##########################################
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)  
    return l


def res_blk2(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
##########################################
#Resnet34 block
##########################################
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split,
                                strides=strides if i == 0 else 1, activation=BNReLU)
                if (strides != 1) and i == 0:
                    l = Conv2D('convshortcut', l, ch[1], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)
    return l
####

def dense_blk(name, l, ch, ksize, count, split=1, padding='valid'):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('blk/' + str(i)):
                x = BNReLU('preact_bna', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], padding=padding, activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], padding=padding, split=split)
                ##
                if padding == 'valid':
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[2] - x_shape[2], 
                                    l_shape[3] - x_shape[3]))

                l = tf.concat([l, x], axis=1)
        l = BNReLU('blk_bna', l)
    return l
####

def conv_block(x, growth_rate, name, freeze):
  x1 = BatchNorm(name + '_0_bn', x, epsilon=1e-5)
  x = tf.nn.relu(x, name=name + '_0_relu')
  
  x1 = Conv2D(name + '_1_conv',  x1, 4 * growth_rate, 1, strides=1, activation=tf.identity, use_bias=False)
  x1 = BatchNorm(name + '_1_bn', x1, epsilon=1e-5)
  x = tf.nn.relu(x1, name=name + '_1_relu')
  x1 = Conv2D(name + '_2_conv', x1, growth_rate, 3, padding='same', use_bias=False)
  x1 = tf.stop_gradient(x1) if freeze else x1
  x =  tf.concat([x, x1], 1, name=name + '_concat')
  return x

def transition_block(x, reduction, name):

  x = BatchNorm(name + '_bn', x, epsilon=1e-5)
  x = tf.nn.relu(x, name=name + '_relu')
  x = Conv2D(name + '_conv', x, int(x.shape[1].value * reduction), 1, use_bias=False)
  x = AvgPooling(name + '_pool', x, 2, strides=2, padding='SAME')
  return x

  
def dense_block(x, freeze, blocks, name):

  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1), freeze=freeze)
    x = tf.stop_gradient(x) if freeze else x
  return x


###############################################################
##################### ENCODERS: ###############################
###############################################################

def encoder50(i, freeze):
    """
    Pre-activated ResNet50 Encoder
    """

    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk('group0', d1, [ 64,  64,  256], [1, 3, 1], 3, strides=1, freeze=freeze)                       
    
    d2 = res_blk('group1', d1, [128, 128,  512], [1, 3, 1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk('group2', d2, [256, 256, 1024], [1, 3, 1], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk('group3', d3, [512, 512, 2048], [1, 3, 1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4
    
    d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
    return [d1, d2, d3, d4]
    
def encoder34(i, freeze):
    """
    Pre-activated ResNet34 Encoder
    """

    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk2('group0', d1, [64,  64], [3,3], 3, strides=1, freeze=freeze)

    d2 = res_blk2('group1', d1, [128, 128], [3,3], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk2('group2', d2, [256, 256], [3,3], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk2('group3', d3, [512, 512], [3,3], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D('conv_bot', d4, 256, 1, padding='same')
    return [d1, d2, d3, d4]

def encoder18(i, freeze):
    """
    Pre-activated ResNet18 Encoder
    """

    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk2('group0', d1, [64,  64], [3,3], 3, strides=1, freeze=freeze)

    d2 = res_blk2('group1', d1, [128, 128], [3,3], 2, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk2('group2', d2, [256, 256], [3,3], 2, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk2('group3', d3, [512, 512], [3,3], 2, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D('conv_bot', d4, 256, 1, padding='same')
    return [d1, d2, d3, d4]

def encoder101(i, freeze):
    """
    Pre-activated ResNet101 Encoder
    """

    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk('group0', d1, [64,  64,256], [1,3,1], 3, strides=1, freeze=freeze)                       
    
    d2 = res_blk('group1', d1, [128, 128,512], [1,3,1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk('group2', d2, [256, 256,1024], [1,3,1], 23, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk('group3', d3, [512, 512,2048], [1,3,1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4
    
    d4 = Conv2D('conv_bot', d4, 1024, 1, padding='same')
    return [d1, d2, d3, d4]

def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype, freeze):
  
  stride = 2 if nr1x1 == 0 else 1
  with tf.variable_scope(name):
    outs = []
    if nr1x1 != 0:
      outs.append(Conv2D('conv1x1', x, nr1x1, 1))
    x2 = Conv2D('conv3x3r', x, nr3x3r, 1)
    outs.append(Conv2D('conv3x3', x2, nr3x3, 3, strides=stride))

    x3 = Conv2D('conv233r', x, nr233r, 1)
    x3 = Conv2D('conv233a', x3, nr233, 3)
    outs.append(Conv2D('conv233b', x3, nr233, 3, strides=stride))

    if pooltype == 'max':
      x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
    else:
      assert pooltype == 'avg'
      x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
    if nrpool != 0:  # pool + passthrough if nrpool == 0
      x4 = Conv2D('poolproj', x4, nrpool, 1)
    outs.append(x4)
    l =  tf.concat(outs, 1, name='concat')
    l = tf.stop_gradient(l) if freeze else l
    return l
    
def inception_encoder(i, freeze):
  
  with argscope(Conv2D, activation=BNReLU, use_bias=False):
    d1 = (LinearWrap(i)
         .Conv2D('conv0', 64, 7, strides=1, padding='valid')
         .Conv2D('conv1', 64, 1)
         .Conv2D('conv2', 192, 3)())
    
    l = inception('incep3a', d1, 64, 64, 64, 64, 96, 32, 'avg', freeze=freeze)
    l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg', freeze=freeze)
    d2 = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max', freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2
    
    l = inception('incep4a', d2, 224, 64, 96, 96, 128, 128, 'avg', freeze=freeze)
    l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg', freeze=freeze)
    l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg', freeze=freeze)
    l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg', freeze=freeze)
    d3 = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max', freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3
    
    l = inception('incep5a', d3, 352, 192, 320, 160, 224, 128, 'avg', freeze=freeze)
    d4 = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max', freeze=freeze)
    d4 = AvgPooling('apool-final', d4, 3, 2, padding='SAME')
    d3= Conv2D('conv-d3-stub',  d3, 1024, 1, strides=1, activation=tf.identity)
    
    return [d1, d2, d3, d4]
  

def densenet_encoder(x, freeze, blocks = [6, 12, 24, 16]):
  
  x = Conv2D('conv1/conv', x, 64, 7, padding='valid', strides=1, activation=BNReLU)
  d1 = dense_block(x, freeze, blocks[0], name='conv2')
  x = transition_block(x, 1, name='pool2')
  d2 = dense_block(x, freeze, blocks[1], name='conv3')
  d2 = tf.stop_gradient(d2) if freeze else d2
  x = transition_block(d2, 0.5, name='pool3')
  d3 = dense_block(x, freeze, blocks[2], name='conv4')
  d3 = tf.stop_gradient(d3) if freeze else d3
  x = transition_block(d3, 0.5, name='pool4')
  d4 = dense_block(x, freeze, blocks[3], name='conv5')
  d4 = tf.stop_gradient(d4) if freeze else d4
  
  return [d1, d2, d3, d4]


def atrous_spatial_pyramid_pooling(x, filters=64):
  
  pad = 'valid'      
  with tf.variable_scope('ASSP_layers'):

    image_level_features = tf.reduce_mean(x, [2, 3], keep_dims=True)
    image_level_features = Conv2D('image_level_features', image_level_features, filters, 1, strides=1, padding=pad, activation=BNReLU) 
    image_level_features = FixedUnPooling('unp_image_level_features', image_level_features, shape=(x.shape[2].value, x.shape[3].value), data_format='channels_first') 

    at_pool1x1   = Conv2D('at_pool1x1', x, filters, 1, strides=1, padding=pad, activation=BNReLU)  
    at_pool3x3_1 = Conv2D('at_pool3x3_1', x, filters, 3, strides=1, padding='same', activation=BNReLU, dilation_rate=6)  
    at_pool3x3_2 = Conv2D('at_pool3x3_2', x, filters, 3, strides=1, padding='same', activation=BNReLU, dilation_rate=12) 
    at_pool3x3_3 = Conv2D('at_pool3x3_3', x, filters, 3, strides=1, padding='same', activation=BNReLU, dilation_rate=18) 

    net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=1)

    net =  Conv2D('at_out', net, filters, 1, strides=1, padding=pad, activation=BNReLU)

    return net


def decoder(name, i, use_assp=True):
    pad = 'same' #Change to valid if non local 
    with tf.variable_scope(name):
        with tf.variable_scope('u3'):
            u3 = upsample2x('rz', i[-1])
            u3_sum = tf.add_n([u3, i[-2]])

            u3 = Conv2D('conva', u3_sum, 256, 5, strides=1, padding=pad)   
            u3 = dense_blk('dense', u3, [128, 32], [1, 5], 8, split=4, padding=pad)
            u3 = Conv2D('convf', u3, i[-3].shape[1].value, 1, strides=1)   
        ####
        with tf.variable_scope('u2'):          
            u2 = upsample2x('rz', u3)
            u2_sum = tf.add_n([u2, i[-3]])

            u2x = Conv2D('conva', u2_sum, 128, 5, strides=1, padding=pad)
            u2 = dense_blk('dense', u2x, [128, 32], [1, 5], 4, split=4, padding=pad)
            u2 = Conv2D('convf', u2, i[-4].shape[1].value, 1, strides=1) 
            #u2 = nonlocal_dot(u2, depth=128, embed=True, softmax=True, maxpool=2, scope='nonlocaldot') # Uncomment for non-local block
        ####
        with tf.variable_scope('u1'):          
            u1 = upsample2x('rz', u2)
            u1_sum = tf.add_n([u1, i[-4]])
            u1 = Conv2D('conva', u1_sum, 64, 5, strides=1, padding='same')
        
        if use_assp:
          u1 = atrous_spatial_pyramid_pooling(u1)
          
    return [u3, u2x, u1]


####
class Model(ModelDesc, Config):
    def __init__(self, freeze=False, encoder_name='default'):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = 'NCHW'
        self.encoder_name = encoder_name

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [None], 'truemap-coded')]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.optimizer(learning_rate=lr)
        return opt

####
class Model_NP_HV(Model):
    def _build_graph(self, inputs):
        
        images, truemap_coded = inputs
        orig_imgs = images

        if hasattr(self, 'type_classification') and self.type_classification:
            true_type = truemap_coded[...,1]
            true_type = tf.cast(true_type, tf.int32)
            true_type = tf.identity(true_type, name='truemap-type')
            one_type  = tf.one_hot(true_type, self.nr_types, axis=-1)
            true_type = tf.expand_dims(true_type, axis=-1)

            true_np = tf.cast(true_type > 0, tf.int32) # ? sanity this
            true_np = tf.identity(true_np, name='truemap-np')
            one_np  = tf.one_hot(tf.squeeze(true_np), 2, axis=-1)
        else:
            true_np = truemap_coded[...,0]
            true_np = tf.cast(true_np, tf.int32)
            true_np = tf.identity(true_np, name='truemap-np')
            one_np  = tf.one_hot(true_np, 2, axis=-1)
            true_np = tf.expand_dims(true_np, axis=-1)

        true_hv = truemap_coded[...,-2:]
        true_hv = tf.identity(true_hv, name='truemap-hv')

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False, # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm, MaxPooling, AvgPooling], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0
            #i = images #IF non local

            ####
            #d = encoder(i, self.freeze)
            if self.encoder_name == 'inception':
              d = inception_encoder(i, self.freeze)
            elif self.encoder_name == 'densenet':
              d = densenet_encoder(i, self.freeze)
            else:
              d = encoder(i, self.freeze)
            d[0] = crop_op(d[0], (184, 184))
            d[1] = crop_op(d[1], (72, 72))

            ####
            np_feat = decoder('np', d, self.use_assp)
            npx = BNReLU('preact_out_np', np_feat[-1])

            hv_feat = decoder('hv', d, self.use_assp)
            hv = BNReLU('preact_out_hv', hv_feat[-1])

            if self.type_classification:
                tp_feat = decoder('tp', d, self.use_assp)
                tp = BNReLU('preact_out_tp', tp_feat[-1])

                # Nuclei Type Pixels (TP)
                logi_class = Conv2D('conv_out_tp', tp, self.nr_types, 1, use_bias=True, activation=tf.identity)
                logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
                soft_class = tf.nn.softmax(logi_class, axis=-1)

            #### Nuclei Pixels (NP)
            logi_np = Conv2D('conv_out_np', npx, 2, 1, use_bias=True, activation=tf.identity)
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[...,1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)

            #### Horizontal-Vertival (HV)
            logi_hv = Conv2D('conv_out_hv', hv, 2, 1, use_bias=True, activation=tf.identity)
            logi_hv = tf.transpose(logi_hv, [0, 2, 3, 1])
            prob_hv = tf.identity(logi_hv, name='predmap-prob-hv')
            pred_hv = tf.identity(logi_hv, name='predmap-hv')
    
            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            if self.type_classification:
                predmap_coded = tf.concat([soft_class, prob_np, pred_hv], axis=-1, name='predmap-coded')
            else:
                predmap_coded = tf.concat([prob_np, pred_hv], axis=-1, name='predmap-coded')
        ####
        def get_gradient_hv(l, h_ch, v_ch):
            def get_sobel_kernel(size):
                assert size % 2 == 1, 'Must be odd, get size=%d' % size

                h_range = np.arange(-size//2+1, size//2+1, dtype=np.float32)
                v_range = np.arange(-size//2+1, size//2+1, dtype=np.float32)
                h, v = np.meshgrid(h_range, v_range)
                kernel_h = h / (h * h + v * v + 1.0e-15)
                kernel_v = v / (h * h + v * v + 1.0e-15)
                return kernel_h, kernel_v            

            mh, mv = get_sobel_kernel(5)
            mh = tf.constant(mh, dtype=tf.float32)
            mv = tf.constant(mv, dtype=tf.float32)

            mh = tf.reshape(mh, [5, 5, 1, 1])
            mv = tf.reshape(mv, [5, 5, 1, 1])
            
            # central difference to get gradient, ignore the boundary problem  
            h = tf.expand_dims(l[...,h_ch], axis=-1)  
            v = tf.expand_dims(l[...,v_ch], axis=-1)  
            dh = tf.nn.conv2d(h, mh, strides=[1, 1, 1, 1], padding='SAME')
            dv = tf.nn.conv2d(v, mv, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.concat([dh, dv], axis=-1)
            return output
        
#######################################################
#                  LOSS FUNCTIONS
#######################################################

#Loss for distances using horizontal/vertical distance to nuclear centroids.

        def loss_mse(true,pred,name=None):
            ###########################
            #Standard MSE:
            ###########################
            loss = pred - true
            loss = tf.reduce_mean(loss * loss, name=name)
            
            return loss
        
        
        def loss_huber(true,pred,name=None):
            ###########################
            #Standard Huber Loss:
            ###########################
            alpha = 0.5
            huber_mse = alpha*(true-pred)**2
            huber_mae = alpha*tf.math.abs(true-pred)
            loss = tf.where((tf.math.abs(true - pred)) <= alpha, huber_mse, huber_mae,name=name)
            
            return tf.reduce_mean(loss)  
        
        
        def loss_huber_smooth(true,pred,name=None):
            ###########################
            #Smoother Huber Loss:
            ###########################
            alpha = 0.5
            delta = 2
            eps = 1e-6
            huber_smooth_mse = alpha*(true-pred)**2
            huber_novel = alpha*tf.math.log((0.5*(1+tf.math.exp(delta*(true-pred)+eps)))**(2/delta))-(true-pred)
            loss = tf.where((tf.math.abs(true - pred)) <= alpha, huber_smooth_mse, huber_novel,name=name)
            return tf.reduce_mean(loss)  
  
    
        def loss_msge(true, pred, focus, name=None):
            ###########################
            #MSE of gradients:
            ###########################
            focus = tf.stack([focus, focus], axis=-1)
            pred_grad = get_gradient_hv(pred, 1, 0)
            true_grad = get_gradient_hv(true, 1, 0)
            loss = pred_grad - true_grad
            loss = focus * (loss * loss)
            # artificial reduce_mean with focus region
            loss = tf.reduce_sum(loss) / (tf.reduce_sum(focus) + 1.0e-8)
            loss = tf.identity(loss, name=name)
            return loss
        
        ####
        if get_current_tower_context().is_training:
            #---- LOSS ----#
            loss = 0
            for term, weight in self.loss_term.items():
                if term == 'mse':
                    term_loss = loss_mse(true_hv, pred_hv, name='loss-mse')
                elif term == 'msge':
                    focus = truemap_coded[...,0]
                    term_loss = loss_msge(true_hv, pred_hv, focus, name='loss-msge')
                elif term == 'bce':
                    term_loss = categorical_crossentropy(soft_np, one_np)
                    term_loss = tf.reduce_mean(term_loss, name='loss-bce')
                elif 'dice' in self.loss_term:
                    term_loss = dice_loss(soft_np[...,0], one_np[...,0]) \
                              + dice_loss(soft_np[...,1], one_np[...,1])
                    term_loss = tf.identity(term_loss, name='loss-dice')
                else:
                    assert False, 'Not support loss term: %s' % term
                add_moving_summary(term_loss)
                loss += term_loss * weight

            if self.type_classification:
                term_loss = categorical_crossentropy(soft_class, one_type)
                term_loss = tf.reduce_mean(term_loss, name='loss-xentropy-class')
                add_moving_summary(term_loss)
                loss = loss + term_loss

                term_loss = 0
                for type_id in range(self.nr_types):
                    term_loss += dice_loss(soft_class[...,type_id], 
                                           one_type[...,type_id])
                term_loss = tf.identity(term_loss, name='loss-dice-class')
                add_moving_summary(term_loss)
                loss = loss + term_loss

            ### combine the loss into single cost function
            self.cost = tf.identity(loss, name='overall-loss')            
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            ### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (190, 190), "NHWC")

            pred_np = colorize(prob_np[...,0], cmap='jet')
            true_np = colorize(true_np[...,0], cmap='jet')
            
            pred_h = colorize(prob_hv[...,0], vmin=-1, vmax=1, cmap='jet')
            pred_v = colorize(prob_hv[...,1], vmin=-1, vmax=1, cmap='jet')
            true_h = colorize(true_hv[...,0], vmin=-1, vmax=1, cmap='jet')
            true_v = colorize(true_hv[...,1], vmin=-1, vmax=1, cmap='jet')

            if not self.type_classification:
                viz = tf.concat([orig_imgs, 
                                pred_h, pred_v, pred_np, 
                                true_h, true_v, true_np], 2)
            else:
                pred_type = tf.transpose(soft_class, (0, 1, 3, 2))
                pred_type = tf.reshape(pred_type, [-1, 80, 80 * self.nr_types])
                true_type = tf.cast(true_type[...,0] / self.nr_classes, tf.float32)
                true_type = colorize(true_type, vmin=0, vmax=1, cmap='jet')
                pred_type = colorize(pred_type, vmin=0, vmax=1, cmap='jet')

                viz = tf.concat([orig_imgs, 
                                pred_h, pred_v, pred_np, pred_type, 
                                true_h, true_v, true_np, true_type,], 2)

            viz = tf.concat([viz[0], viz[-1]], axis=0)
            viz = tf.expand_dims(viz, axis=0)
            #print(viz.name)
            viz_size = (240, 160)
            tf.summary.image('output', viz, max_outputs=1)

        return
####

class Model_NP_DIST(Model):
    def _build_graph(self, inputs):
       
        images, truemap_coded = inputs

        orig_imgs = images

        true_np = truemap_coded[...,0]
        true_np = tf.cast(true_np, tf.int32)
        true_np = tf.identity(true_np, name='truemap-np')
        one_np  = tf.one_hot(true_np, 2, axis=-1)
        true_np = tf.expand_dims(true_np, axis=-1)

        true_dist = truemap_coded[...,1:]
        true_dist = tf.identity(true_dist, name='truemap-dist')

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False, # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = encoder(i, self.freeze)
            d[0] = crop_op(d[0], (184, 184))
            d[1] = crop_op(d[1], (72, 72))

            ####
            np_feat = decoder('np', d)
            np = BNReLU('preact_out_np', np_feat[-1])

            dist_feat = decoder('dst', d)
            dist = BNReLU('preact_out_dist', dist_feat[-1])

            #### Initialise weights prior distn
            logi_np = Conv2D('conv_out_np', npx, 2, 1, use_bias=True, activation=tf.identity, bias_initializer=tf.constant_initializer(value=-np.log((1 - 0.05) / 0.05)))
            
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.sigmoid(logi_np) #Change to softmax if not using focal 
            prob_np = tf.identity(soft_np[...,1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)
            pred_np = tf.argmax(soft_np, axis=-1, name='predmap-np')
            pred_np = tf.expand_dims(tf.cast(pred_np, tf.float32), axis=-1)

            ####
            logi_dist = Conv2D('conv_out_dist', dist, 1, 1, use_bias=True, activation=tf.identity)
            logi_dist = tf.transpose(logi_dist, [0, 2, 3, 1])
            prob_dist = tf.identity(logi_dist, name='predmap-prob-dist')
            pred_dist = tf.identity(logi_dist, name='predmap-dist')

            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([prob_np, pred_dist], axis=-1, name='predmap-coded')
        ####

        ####
        if get_current_tower_context().is_training:
            ######## LOSS
            ### Distance regression loss
            loss_mse = pred_dist - true_dist
            loss_mse = loss_mse * loss_mse
            loss_mse = tf.reduce_mean(loss_mse, name='loss-mse')
            add_moving_summary(loss_mse)   

            ### Nuclei Blob classification loss
            loss_bce = categorical_crossentropy(soft_np, one_np)
            loss_bce = tf.reduce_mean(loss_bce, name='loss-bce')
            add_moving_summary(loss_bce)

            ### combine the loss into single cost function
            self.cost = tf.identity(loss_mse + loss_bce, name='overall-loss')            
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (190, 190), "NHWC")

            pred_np = colorize(prob_np[...,0], cmap='jet')
            true_np = colorize(true_np[...,0], cmap='jet')

            pred_dist = colorize(prob_dist[...,0], cmap='jet')
            true_dist = colorize(true_dist[...,0], cmap='jet')

            viz = tf.concat([orig_imgs, 
                            true_np, pred_np, 
                            true_dist, pred_dist,], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return

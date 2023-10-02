from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenatev
import tensorflow as tf


def perceptual_loss(y_true, y_pred):  # Note the parameter order

    mod = VGG19(include_top=False, weights='imagenet')
    pred = concatenate([y_pred, y_pred, y_pred])
    true = concatenate([y_true, y_true, y_true])
    vggmodel_1 = Model(inputs=mod.input, outputs=mod.get_layer('block1_pool').output)
    vggmodel_2 = Model(inputs=mod.input, outputs=mod.get_layer('block2_pool').output)
    vggmodel_3 = Model(inputs=mod.input, outputs=mod.get_layer('block3_pool').output)
    vggmodel_4 = Model(inputs=mod.input, outputs=mod.get_layer('block4_pool').output)
    vggmodel_5 = Model(inputs=mod.input, outputs=mod.get_layer('block5_pool').output)

    f_p_1 = vggmodel_1(pred)
    f_t_1 = vggmodel_1(true)
    f_p_2 = vggmodel_2(pred)
    f_t_2 = vggmodel_2(true)
    f_p_3 = vggmodel_3(pred)
    f_t_3 = vggmodel_3(true)
    f_p_4 = vggmodel_4(pred)
    f_t_4 = vggmodel_4(true)
    f_p_5 = vggmodel_5(pred)
    f_t_5 = vggmodel_5(true)

    pl_1 = tf.math.reduce_mean(tf.math.square(f_p_1 - f_t_1))
    pl_2 = tf.math.reduce_mean(tf.math.square(f_p_2 - f_t_2))
    pl_3 = tf.math.reduce_mean(tf.math.square(f_p_3 - f_t_3))
    pl_4 = tf.math.reduce_mean(tf.math.square(f_p_4 - f_t_4))
    pl_5 = tf.math.reduce_mean(tf.math.square(f_p_5 - f_t_5))

    total_pl = pl_1 * (1/16) + pl_2 * (1/8) + pl_3 * (1/4) + pl_4 * (1/2) + pl_5 * (1/1)

    return total_pl

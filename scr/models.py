import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Maximum, Add, Input, Multiply, Subtract, Conv2D, Conv2DTranspose, UpSampling2D, concatenate, MaxPooling2D, Lambda, Reshape, LeakyReLU, BatchNormalization, Dense, Dropout, Activation
from attention import parameter_attention
from fusion import MixedFusion_block_d, MixedFusion_block_u, MixedFusion_block_0
from losses import perceptual_loss



# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X11, X12, X13, X2 = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # scale to [-1,1]
    X11 = (X11 - (3000 / 2)) / (3000 / 2)
    X12 = (X12 - (3000 / 2)) / (3000 / 2)
    X13 = (X13 - (3000 / 2)) / (3000 / 2)
    X2 = (X2 - (3000 / 2)) / (3000 / 2)
    return [X11, X12, X13, X2]
	
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image1 = Input(shape=image_shape)
    in_src_image2 = Input(shape=image_shape)
    in_src_image3 = Input(shape=image_shape)

    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image1, in_src_image2, in_src_image3, in_target_image]) #!!!
    #merged = in_target_image
    # C64
    d = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (3, 3), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (3, 3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image1, in_src_image2, in_src_image3, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[1])
    return model

def maxpooling():
    pooling = MaxPooling2D((2, 2), strides=(2, 2), padding='same')
    return pooling

def define_reconstruction(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input, MODALITY 1
    in_src_image1 = Input(shape=image_shape)
    # C64
    d1_0 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image1)
    d1_0 = BatchNormalization()(d1_0)
    d1_0 = LeakyReLU(alpha=0.2)(d1_0)
    d1_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_0)
    d1_0 = BatchNormalization()(d1_0)
    d2_0 = LeakyReLU(alpha=0.2)(d1_0)
    # C128
    d3_0 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_0)
    d3_0 = BatchNormalization()(d3_0)
    d3_0 = LeakyReLU(alpha=0.2)(d3_0)
    d3_0 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_0)
    d3_0 = BatchNormalization()(d3_0)
    d4_0 = LeakyReLU(alpha=0.2)(d3_0)
    # C256
    d5_0 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_0)
    d5_0 = BatchNormalization()(d5_0)
    d5_0 = LeakyReLU(alpha=0.2)(d5_0)
    d5_0 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_0)
    d5_0 = BatchNormalization()(d5_0)
    d6_0 = LeakyReLU(alpha=0.2)(d5_0)
    # C512
    d7_0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_0)
    d7_0 = BatchNormalization()(d7_0)
    d7_0 = LeakyReLU(alpha=0.2)(d7_0)
    d7_0 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_0)
    d7_0 = BatchNormalization()(d7_0)
    d8_0 = LeakyReLU(alpha=0.2)(d7_0)

    # bottleneck, no batch norm and relu
    b1_0 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_0)
    b2_0 = Activation('relu')(b1_0)

    # upsampling
    # c512
    u1_0 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_0)
    u2_0 = BatchNormalization()(u1_0)
    u3_0 = Activation('relu')(u2_0)
    # c256
    u4_0 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_0)
    u5_0 = BatchNormalization()(u4_0)
    u6_0 = Activation('relu')(u5_0)
    # c128
    u7_0 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_0)
    u8_0 = BatchNormalization()(u7_0)
    u9_0 = Activation('relu')(u8_0)
    # c64
    u10_0 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_0)
    u11_0 = BatchNormalization()(u10_0)
    u12_0 = Activation('relu')(u11_0)

    u13_0 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_0)
    out_image_0 = Activation('tanh')(u13_0)

    # source image input, MODALITY 2
    in_src_image2 = Input(shape=image_shape)
    # C64
    d1_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image2)
    d1_1 = BatchNormalization()(d1_1)
    d1_1 = LeakyReLU(alpha=0.2)(d1_1)
    d1_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_1)
    d1_1 = BatchNormalization()(d1_1)
    d2_1 = LeakyReLU(alpha=0.2)(d1_1)
    # C128
    d3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_1)
    d3_1 = BatchNormalization()(d3_1)
    d3_1 = LeakyReLU(alpha=0.2)(d3_1)
    d3_1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_1)
    d3_1 = BatchNormalization()(d3_1)
    d4_1 = LeakyReLU(alpha=0.2)(d3_1)
    # C256
    d5_1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_1)
    d5_1 = BatchNormalization()(d5_1)
    d5_1 = LeakyReLU(alpha=0.2)(d5_1)
    d5_1 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_1)
    d5_1 = BatchNormalization()(d5_1)
    d6_1 = LeakyReLU(alpha=0.2)(d5_1)
    # C512
    d7_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_1)
    d7_1 = BatchNormalization()(d7_1)
    d7_1 = LeakyReLU(alpha=0.2)(d7_1)
    d7_1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_1)
    d7_1 = BatchNormalization()(d7_1)
    d8_1 = LeakyReLU(alpha=0.2)(d7_1)

    # bottleneck, no batch norm and relu
    b1_1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_1)
    b2_1 = Activation('relu')(b1_1)

    # upsampling
    # c512
    u1_1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_1)
    u2_1 = BatchNormalization()(u1_1)
    u3_1 = Activation('relu')(u2_1)
    # c256
    u4_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_1)
    u5_1 = BatchNormalization()(u4_1)
    u6_1 = Activation('relu')(u5_1)
    # c128
    u7_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_1)
    u8_1 = BatchNormalization()(u7_1)
    u9_1 = Activation('relu')(u8_1)
    # c64
    u10_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_1)
    u11_1 = BatchNormalization()(u10_1)
    u12_1 = Activation('relu')(u11_1)

    u13_1 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_1)
    out_image_1 = Activation('tanh')(u13_1)

    # source image input, MODALITY 3
    in_src_image3 = Input(shape=image_shape)
    # C64
    d1_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image3)
    d1_2 = BatchNormalization()(d1_2)
    d1_2 = LeakyReLU(alpha=0.2)(d1_2)
    d1_2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_2)
    d1_2 = BatchNormalization()(d1_2)
    d2_2 = LeakyReLU(alpha=0.2)(d1_2)
    # C128
    d3_2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_2)
    d3_2 = BatchNormalization()(d3_2)
    d3_2 = LeakyReLU(alpha=0.2)(d3_2)
    d3_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_2)
    d3_2 = BatchNormalization()(d3_2)
    d4_2 = LeakyReLU(alpha=0.2)(d3_2)
    # C256
    d5_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_2)
    d5_2 = BatchNormalization()(d5_2)
    d5_2 = LeakyReLU(alpha=0.2)(d5_2)
    d5_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_2)
    d5_2 = BatchNormalization()(d5_2)
    d6_2 = LeakyReLU(alpha=0.2)(d5_2)
    # C512
    d7_2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_2)
    d7_2 = BatchNormalization()(d7_2)
    d7_2 = LeakyReLU(alpha=0.2)(d7_2)
    d7_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_2)
    d7_2 = BatchNormalization()(d7_2)
    d8_2 = LeakyReLU(alpha=0.2)(d7_2)

    # bottleneck, no batch norm and relu
    b1_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_2)
    b2_2 = Activation('relu')(b1_2)

    # upsampling
    # c512
    u1_2 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_2)
    u2_2 = BatchNormalization()(u1_2)
    u3_2 = Activation('relu')(u2_2)
    # c256
    u4_2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_2)
    u5_2 = BatchNormalization()(u4_2)
    u6_2 = Activation('relu')(u5_2)
    # c128
    u7_2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_2)
    u8_2 = BatchNormalization()(u7_2)
    u9_2 = Activation('relu')(u8_2)
    # c64
    u10_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_2)
    u11_2 = BatchNormalization()(u10_2)
    u12_2 = Activation('relu')(u11_2)

    u13_2 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_2)
    out_image_2 = Activation('tanh')(u13_2)

    # Fusion
    g_d1 = MixedFusion_block_0(d2_0, d2_1, d2_2)
    # g_d1 = BatchNormalization()(g_d1)
    g_d11 = maxpooling()(g_d1)
    g_d2 = MixedFusion_block_d(d4_0, d4_1, d4_2, g_d11)
    # g_d2 = BatchNormalization()(g_d2)
    g_d21 = maxpooling()(g_d2)
    g_d3 = MixedFusion_block_d(d6_0, d6_1, d6_2, g_d21)
    # g_d3 = BatchNormalization()(g_d3)
    g_d31 = maxpooling()(g_d3)
    g_d4 = MixedFusion_block_d(d8_0, d8_1, d8_2, g_d31)

    # bottleneck, no batch norm and relu
    b_s1 = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g_d4)
    b_s1 = Activation('relu')(b_s1)
    b_s2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(b_s1)
    b_s2 = Activation('relu')(b_s2)

    g_u1 = MixedFusion_block_u(u3_0, u3_1, u3_2, b_s2, g_d4)
    # g_u1 = BatchNormalization()(g_u1)
    g_u1 = UpSampling2D(size=(2, 2))(g_u1)
    g_u2 = MixedFusion_block_u(u6_0, u6_1, u6_2, g_u1, g_d3)
    # g_u2 = BatchNormalization()(g_u2)
    g_u2 = UpSampling2D(size=(2, 2))(g_u2)
    g_u3 = MixedFusion_block_u(u9_0, u9_1, u9_2, g_u2, g_d2)
    # g_u3 = BatchNormalization()(g_u3)
    g_u3 = UpSampling2D(size=(2, 2))(g_u3)
    g_u4 = MixedFusion_block_u(u12_0, u12_1, u12_2, g_u3, g_d1)
    # g_u4 = BatchNormalization()(g_u4)
    g_u4 = UpSampling2D(size=(2, 2))(g_u4)
    g_ue = Conv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g_u4)
    gen_image = Activation('tanh')(g_ue)

    model = Model([in_src_image1, in_src_image2, in_src_image3], [out_image_0, out_image_1, out_image_2, gen_image])

    return model

def define_gan(g_model, d_model, image_shape):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	in_src1 = Input(shape=image_shape)
	in_src2 = Input(shape=image_shape)
	in_src3 = Input(shape=image_shape)
	out_src1, out_src2, out_src3, gen_out  = g_model([in_src1, in_src2, in_src3])
	dis_out = d_model([in_src1, in_src2, in_src3, gen_out])
	model = Model([in_src1, in_src2, in_src3], [dis_out, out_src1, out_src2, out_src3, gen_out, gen_out]) 
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae', 'mae', 'mae', 'mae', perceptual_loss], optimizer=opt, loss_weights=[5,25,25,25,100,200])
	return model

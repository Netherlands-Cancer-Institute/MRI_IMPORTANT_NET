import os
from numpy import load, zeros, ones
import numpy as np
import random
from numpy.random import randint
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from models import load_real_samples, define_discriminator, define_reconstruction, define_gan
from config import TRAIN_FILEPATH, VALID_FILEPATH, SAVE_PATH, SEED, IMAGE_SHAPE, TRAINING_EPOCH, NUMBER_BATCH

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
	
filepath=TRAIN_FILEPATH
files=os.listdir(filepath)
files.sort()

def generate_real(files, n_samples, patch_shape1, patch_shape2, bat_per_epo, i):

	# retrieve selected images
	n = int((i)/bat_per_epo)
	name=files[(i-n*bat_per_epo)]
	path = filepath + name
	data=load_real_samples(path)
	X1, X2, X3, X= data
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape1, patch_shape2, 1))
	return [X1, X2, X3, X], y

def generate_real_test(n_samples, patch_shape1, patch_shape2):
	path = VALID_FILEPATH
	print(path)
	data=load_real_samples(path)
	X1, X2, X3, X= data
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape1, patch_shape2, 1))
	return [X1, X2, X3, X], y

def generate_real_show(files, n_samples, patch_shape1, patch_shape2, bat_per_epo, i):
    # retrieve selected images
    n = int((i)/bat_per_epo)
    name=files[(i-n*bat_per_epo)]
    path = filepath + name
    data=load_real_samples(path)
    X1, X2, X3, X= data
    ix = randint(0, X.shape[0], n_samples)
    X11, X12, X13, X22= X1[ix], X2[ix], X3[ix], X[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape1, patch_shape2, 1))
    return [X11, X12, X13, X22], y

def generate_fake_samples(g_model, samples1, samples2, samples3, patch_shape1, patch_shape2):
    # generate fake instance
    _, _, _, X = g_model.predict([samples1, samples2, samples3])  # !!!
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape1, patch_shape2, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, files, bat_per_epo, n_samples=6):
    # select a sample of input images
    [X_realA1, X_realA2, X_realA3, X_realB], _ = generate_real_show(files, n_samples, 1, 1, bat_per_epo, step)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA1, X_realA2, X_realA3, 1, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA1 = (X_realA1 + 1) / 2.0
    X_realA2 = (X_realA2 + 1) / 2.0
    X_realA3 = (X_realA3 + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    X_realAA1 = X_realA1[:, :, :, 0]
    X_realAA2 = X_realA2[:, :, :, 0]
    X_realAA3 = X_realA3[:, :, :, 0]
    X_realBB = X_realB[:, :, :, 0]
    X_fakeBB = X_fakeB[:, :, :, 0]
    # plot real source images1
    for i in range(n_samples):
        pyplot.subplot(5, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realAA1[i], cmap="gray")
    # plot real source images2
    for i in range(n_samples):
        pyplot.subplot(5, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_realAA2[i], cmap="gray")
    # plot real source images2
    for i in range(n_samples):
        pyplot.subplot(5, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realAA3[i], cmap="gray")
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(5, n_samples, 1 + n_samples * 3 + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeBB[i], cmap="gray")
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(5, n_samples, 1 + n_samples * 4 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realBB[i], cmap="gray")
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(SAVE_PATH + filename1, dpi=300)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(SAVE_PATH + filename2)
    print('&gt;Saved: %s and %s' % (filename1, filename2))
  

def train(d_model, g_model, gan_model, files, n_epochs=TRAINING_EPOCH, n_batch=NUMBER_BATCH):
    SSIM = []
    PSNR = []
    MSE = []
    SSIMttt = []
    PSNRttt = []
    MSEttt = []
    n_patch1 = d_model.output_shape[1]
    n_patch2 = d_model.output_shape[2]
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(files) /(n_batch/NUMBER_BATCH))
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA1, X_realA2, X_realA3, X_realB], y_real = generate_real(files, n_batch, n_patch1, n_patch2, bat_per_epo, i)  # !!!
        # generate a batch of fake samples  generate_real(files, n_samples, patch_shape1, patch_shape2, bat_per_epo, i)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA1, X_realA2, X_realA3, n_patch1, n_patch2)  # !!!

        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA1, X_realA2, X_realA3, X_realB], y_real)  # !!!
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA1, X_realA2, X_realA3, X_fakeB], y_fake)  # !!!
        # update the generator
        g_loss, d, r1, r2, r3, gl, pl = gan_model.train_on_batch([X_realA1, X_realA2, X_realA3], [y_real, X_realA1, X_realA2, X_realA3, X_realB, X_realB])
        print('&gt;%d, &train: d1[%.3f] d2[%.3f] g[%.3f] d[%.3f] r1[%.3f] r2[%.3f] r3[%.3f] gl[%.3f] pl[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss, d, r1, r2, r3, gl, pl))
        # summarize model performance (step, g_model, files, bat_per_epo, n_samples=6)
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, files,bat_per_epo)
        K.clear_session()
        if (i + 1) % (bat_per_epo * 1) == 0:
            epoch_train = (i + 1) / (bat_per_epo * 1)
            [X_realA1, X_realA2, X_realA3, X_realB], y_real = generate_real(files, n_batch, n_patch1, n_patch2, bat_per_epo, i)
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA1, X_realA2, X_realA3, n_patch1, n_patch2)
            [X_realA_t1, X_realA_t2, X_realA_t3, X_realB_t], y_real_t = generate_real_test(n_batch, n_patch1,n_patch2)
            X_fakeB_t, y_fake_t = generate_fake_samples(g_model, X_realA_t1, X_realA_t2, X_realA_t3, n_patch1,n_patch2)
            SSIMv = structural_similarity(X_fakeB.squeeze(), X_realB.squeeze(), multichannel=True)
            PSNRv = peak_signal_noise_ratio(X_fakeB.squeeze(), X_realB.squeeze())
            MSEv = mean_squared_error(X_fakeB.squeeze(), X_realB.squeeze())
            SSIMt = structural_similarity(X_fakeB_t.squeeze(), X_realB_t.squeeze(), multichannel=True)
            PSNRt = peak_signal_noise_ratio(X_fakeB_t.squeeze(), X_realB_t.squeeze())
            MSEt = mean_squared_error(X_fakeB_t.squeeze(), X_realB_t.squeeze())
            print('&epoch %d, &gt %d, &train: d1[%.3f] d2[%.3f] g[%.3f] d[%.3f] r1[%.3f] r2[%.3f] r3[%.3f] gl[%.3f] pl[%.3f] &val: ssim_v[%.3f] psnr_v[%.3f] mse_v[%.5f], &test:ssim_t[%.3f] psnr_t[%.3f] mse_t[%.5f]' % (epoch_train, i + 1, d_loss1, d_loss2, g_loss, d, r1, r2, r3, gl, pl, SSIMv, PSNRv, MSEv, SSIMt, PSNRt, MSEt))

            SSIM.append(SSIMv)
            PSNR.append(PSNRv)
            MSE.append(MSEv)
            SSIMttt.append(SSIMt)
            PSNRttt.append(PSNRt)
            MSEttt.append(MSEt)
            K.clear_session()

    a = SSIM
    b = PSNR
    c = MSE
    d = SSIMttt
    e = PSNRttt
    f = MSEttt
    dataframe = pd.DataFrame({'SSIM': a, 'PSNR': b, 'MSE': c, 'SSIMttt': d, 'PSNRttt': e, 'MSEttt': f})
    dataframe.to_csv(SAVE_PATH + "indicators.csv", index=False, sep=',')

image_shape = IMAGE_SHAPE

# define the models
d_model = define_discriminator(image_shape)
g_model = define_reconstruction(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# train model
if __name__ == '__main__':
	train(d_model, g_model, gan_model, files)

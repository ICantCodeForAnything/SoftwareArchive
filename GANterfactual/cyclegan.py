from __future__ import print_function, division

import datetime
import os
import sys
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from skimage.transform import resize

from keras.layers import Input, Dropout, Concatenate
from keras.models import Model
# from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from tensorflow.keras.optimizers.legacy import Adam

from classifier import load_classifier
from dataloader import DataLoader
from discriminator import build_discriminator
from generator import build_generator

class CycleGAN():
    def __init__(self, model):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3 if model != 'alexnet' else 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.model = model
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        self.d_N = None
        self.d_P = None
        self.g_NP = None
        self.g_PN = None
        self.combined = None
        self.classifier = None

    def construct(self, classifier_path=None, classifier_weight=None):
        # Build the discriminators
        self.d_N = build_discriminator(self.img_shape, self.df)
        self.d_P = build_discriminator(self.img_shape, self.df)

        # Build the generators
        self.g_NP = build_generator(self.img_shape, self.gf, self.channels)
        self.g_PN = build_generator(self.img_shape, self.gf, self.channels)

        self.build_combined(classifier_path, classifier_weight)

    def load_existing(self, cyclegan_folder, classifier_path=None, classifier_weight=None):
        custom_objects = {"InstanceNormalization": InstanceNormalization}

        # Load discriminators from disk
        self.d_N = keras.models.load_model(os.path.join(cyclegan_folder, f'{self.model}_discriminator_n.h5'),
                                           custom_objects=custom_objects)
        self.d_N._name = "d_N"
        self.d_P = keras.models.load_model(os.path.join(cyclegan_folder, f'{self.model}_discriminator_p.h5'),
                                           custom_objects=custom_objects)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = keras.models.load_model(os.path.join(cyclegan_folder, f'{self.model}_generator_np.h5'),
                                            custom_objects=custom_objects)
        self.g_NP._name = "g_NP"
        self.g_PN = keras.models.load_model(os.path.join(cyclegan_folder, f'{self.model}_generator_pn.h5'),
                                            custom_objects=custom_objects)
        self.g_PN._name = "g_PN"

        self.build_combined(classifier_path, classifier_weight)

    def save(self, cyclegan_folder):
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d_N.save(os.path.join(cyclegan_folder, f'{self.model}_discriminator_n.h5'))
        self.d_P.save(os.path.join(cyclegan_folder, f'{self.model}_discriminator_p.h5'))

        # Save generators to disk
        self.g_NP.save(os.path.join(cyclegan_folder, f'{self.model}_generator_np.h5'))
        self.g_PN.save(os.path.join(cyclegan_folder, f'{self.model}_generator_pn.h5'))

    def build_combined(self, classifier_path=None, classifier_weight=None):
        optimizer = Adam(0.0002, 0.5)

        self.d_N.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_P.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Input images from both domains
        img_N = Input(shape=self.img_shape)
        img_P = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)
        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)
        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)

        # For the combined model we will only train the generators
        self.d_N.trainable = False
        self.d_P.trainable = False

        # Discriminators determines validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        if classifier_path is not None and os.path.isfile(classifier_path):
            self.classifier = load_classifier(classifier_path, self.img_shape)
            self.classifier._name = "classifier"
            self.classifier.trainable = False

            class_N_loss = self.classifier(fake_N)
            class_P_loss = self.classifier(fake_P)

            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           class_N_loss, class_P_loss,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                classifier_weight, classifier_weight,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer)

        else:
            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer)

    def train(self, dataset_name, epochs, batch_size=1, train_N="normal", train_P="not_normal", print_interval=100,
              sample_interval=1000):

        # Configure data loader
        data_loader = DataLoader(dataset_name=dataset_name, img_res=(self.img_rows, self.img_cols), model=self.model)

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        class_N = np.stack([np.ones(batch_size), np.zeros(batch_size)]).T
        class_P = np.stack([np.zeros(batch_size), np.ones(batch_size)]).T

        for epoch in range(epochs):
            for batch_i, (imgs_N, imgs_P) in enumerate(data_loader.load_batch(train_N, train_P, batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------
                
                # Translate images to opposite domain
                fake_P = self.g_NP.predict(imgs_N)
                fake_N = self.g_PN.predict(imgs_P)

                # Train the discriminators (original images = real / translated = Fake)
                dN_loss_real = self.d_N.train_on_batch(imgs_N, valid)
                dN_loss_fake = self.d_N.train_on_batch(fake_N, fake)
                dN_loss = 0.5 * np.add(dN_loss_real, dN_loss_fake)

                dP_loss_real = self.d_P.train_on_batch(imgs_P, valid)
                dP_loss_fake = self.d_P.train_on_batch(fake_P, fake)
                dP_loss = 0.5 * np.add(dP_loss_real, dP_loss_fake)

                # Total disciminator Adam
                d_loss = 0.5 * np.add(dN_loss, dP_loss)
            
                # ------------------
                #  Train Generators
                # ------------------

                if self.classifier is not None:
                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           class_N, class_P,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])
                else:
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])

                elapsed_time = datetime.datetime.now() - start_time

                if self.classifier is not None:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, classifier_N: {g_loss[3]:.5f}, classifier_P: {g_loss[4]:.5f}, " \
                                   f"recon: {np.mean(g_loss[5:7]):.5f}, id: {np.mean(g_loss[7:9]):.5f}] " \
                                   f"time: {elapsed_time}"
                else:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, recon: {np.mean(g_loss[3:5]):.5f}, id: {np.mean(g_loss[5:7]):.5f}] " \
                                   f"time: {elapsed_time}"

                # Plot the progress
                if batch_i % print_interval == 0:
                    print(progress_str)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_N[0], imgs_P[0])

            # Comment this in if you want to save checkpoints:
            self.save(os.path.join('..','models','GANterfactual','ep_' + str(epoch)))

    def sample_images(self, epoch, batch_i, testN, testP):
        os.makedirs(f'{model}_images', exist_ok=True)
        r, c = 2, 3

        img_N = testN[np.newaxis, :, :, :]
        img_P = testP[np.newaxis, :, :, :]

        # Translate images to the other domain
        fake_P = self.g_NP.predict(img_N)
        fake_N = self.g_PN.predict(img_P)
        # Translate back to original domain
        reconstr_N = self.g_PN.predict(fake_P)
        reconstr_P = self.g_NP.predict(fake_N)

        imgs = [img_N, fake_P, reconstr_N, img_P, fake_N, reconstr_P]
        classification = [['normal', 'not_normal'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

        gen_imgs = np.concatenate(imgs)
        correct_classification = ['normal', 'not_normal', 'normal', 'not_normal', 'normal', 'not_normal']

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(15, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray', vmin=0, vmax=1)
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{model}_images/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def predict(self, original_in_path, translated_out_path, reconstructed_out_path, force_original_aspect_ratio=False, normal_class=True):
        assert (self.classifier is not None)
        data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        original = data_loader.load_single(original_in_path)
        original = original.reshape(1, original.shape[0], original.shape[1], original.shape[2])

        pred_original = int(np.argmax(self.classifier.predict(original)))
        if normal_class:
            translated = self.g_NP.predict(original)
            reconstructed = self.g_PN.predict(translated)
        else:
            translated = self.g_PN.predict(original)
            reconstructed = self.g_NP.predict(translated)

        pred_translated = int(np.argmax(self.classifier.predict(translated)))
        pred_reconstructed = int(np.argmax(self.classifier.predict(reconstructed)))

        if force_original_aspect_ratio:
            orig_no_res = keras.preprocessing.image.load_img(original_in_path)
            translated = resize(translated[0], (orig_no_res.height, orig_no_res.width))
            reconstructed = resize(reconstructed[0], (orig_no_res.height, orig_no_res.width))
        else:
            translated = translated[0]
            reconstructed = reconstructed[0]

        data_loader.save_single(translated, translated_out_path)
        data_loader.save_single(reconstructed, reconstructed_out_path)
        
        def calculate_mae(actual, predicted):
            # Calculate the absolute differences
            absolute_errors = np.abs(actual - predicted)
            # Calculate the mean of these differences
            mae = np.mean(absolute_errors)
            return mae

        mae = calculate_mae(original, reconstructed)
        
        print(mae)

        return [pred_original, pred_translated, pred_reconstructed, mae]

    def predict_without_saving(self, original_in_path, force_original_aspect_ratio=False, normal_class=True):
        assert (self.classifier is not None)
        data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        original = data_loader.load_single(original_in_path)
        original = original.reshape(1, original.shape[0], original.shape[1], original.shape[2])

        pred_original = int(np.argmax(self.classifier.predict(original)))
        if normal_class:
            translated = self.g_NP.predict(original)
            reconstructed = self.g_PN.predict(translated)
        else:
            translated = self.g_PN.predict(original)
            reconstructed = self.g_NP.predict(translated)

        pred_translated = int(np.argmax(self.classifier.predict(translated)))
        pred_reconstructed = int(np.argmax(self.classifier.predict(reconstructed)))
        
        def calculate_mae(actual, predicted):
            # Calculate the absolute differences
            absolute_errors = np.abs(actual - predicted)
            # Calculate the mean of these differences
            mae = np.mean(absolute_errors)
            return mae
        
        mae = calculate_mae(original, reconstructed)
        
        return [pred_original, pred_translated, pred_reconstructed, mae]


def get_first_n_images(folder_path, n = 500):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter the list to only include image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [file for file in files if file.endswith(tuple(image_extensions))]
    
    # Take the first n image files
    first_n_images = image_files[:n]
    
    # Get the full paths of the selected images
    image_paths = [os.path.join(folder_path, image) for image in first_n_images]
    
    return image_paths

if __name__ == '__main__':
    if sys.argv[1] is None:
        print('You did not specify a model to load')
        exit()
    model = sys.argv[1]
    gan = CycleGAN(model)
    if len(sys.argv) >= 3 and sys.argv[2] == 'predict':
        print('testing')
        gan.load_existing(
            cyclegan_folder=os.path.join('..', 'models', 'GANterfactual'), 
            classifier_path=os.path.join('..', 'models', 'classifier', f'{model}_model.h5'), 
            classifier_weight=1
        )
        gan.classifier.trainable = False
        gan.d_N.trainable = False
        gan.d_P.trainable = False
        gan.g_PN.trainable = False
        gan.g_NP.trainable = False
        class_folders = [f.name for f in os.scandir(os.path.join('..', 'data', 'test')) if f.is_dir()]
        if class_folders[0] == 'normal':
            normal_path = os.path.join('..', 'data', 'test', class_folders[0])
            not_normal_path = os.path.join('..', 'data', 'test', class_folders[1])
        else:
            normal_path = os.path.join('..', 'data', 'test', class_folders[1])
            not_normal_path = os.path.join('..', 'data', 'test', class_folders[0])
        
        normal_images_to_test = get_first_n_images(normal_path)
        not_normal_images_to_test = get_first_n_images(not_normal_path)
        
        # Save the generated counterfactuals
        if len(sys.argv) == 4 and sys.argv[3] == 'save':
            os.makedirs(f'./generated_images/translated/{gan.model}', exist_ok=True)
            os.makedirs(f'./generated_images/recon/{gan.model}', exist_ok=True)
            for normal_image_path in normal_images_to_test:
                image_name = os.path.basename(normal_image_path)
                result = gan.predict(normal_image_path, os.path.join('.', 'generated_images', 'translated', gan.model, f'{gan.model}_translated_{image_name}'), os.path.join('.', 'generated_images', 'recon', gan.model, f'{gan.model}_recon_{image_name}'))
                print(f'image: {image_name}, original class: 1, result: {result}')
            for not_normal_image_path in not_normal_images_to_test:
               image_name = os.path.basename(not_normal_image_path)
               result = gan.predict(not_normal_image_path, os.path.join('.', 'generated_images', 'translated', gan.model, f'{gan.model}_translated_{image_name}'), os.path.join('.', 'generated_images', 'recon', gan.model, f'{gan.model}_recon_{image_name}'))
               print(f'image: {image_name}, original class: 0, result: {result}')
        else:
            normal_results = [0, 0, 0]
            mean_reconstruction_loss = 0
            for normal_image_path in normal_images_to_test:
                result = gan.predict_without_saving(normal_image_path)
                if result[0] == 0:
                    normal_results[0] += 1
                if result[1] == 1:
                    normal_results[1] += 1
                if result[2] == 0:
                    normal_results[2] += 1
                mean_reconstruction_loss += result[3]
        
            not_normal_results = [0, 0, 0]
            mean_not_normal_reconstruction_loss = 0
            for not_normal_image_path in not_normal_images_to_test:
                result = gan.predict_without_saving(not_normal_image_path, normal_class=False)
                if result[0] == 1:
                    not_normal_results[0] += 1
                if result[1] == 0:
                    not_normal_results[1] += 1
                if result[2] == 1:
                    not_normal_results[2] += 1
                mean_not_normal_reconstruction_loss += result[3]
            print(f'num normal images: {len(normal_images_to_test)}: results: {normal_results}, recons loss: {mean_reconstruction_loss / len(normal_images_to_test)} ')
            print(f'num not normal images: {len(not_normal_images_to_test)}: results: {not_normal_results}, recons loss: {mean_not_normal_reconstruction_loss / len(normal_images_to_test)} ')
    else:
        print('training')
        gan.construct(classifier_path=os.path.join('..', 'models', 'classifier', f'{model}_model.h5'), classifier_weight=1)
        # gan.load_existing(cyclegan_folder=os.path.join('..', 'models', 'GANterfactual'), classifier_path=os.path.join('..', 'models', 'classifier', 'model.h5'), classifier_weight=1)
        gan.train(dataset_name=os.path.join("..","cyclegan_data"), epochs=20, batch_size=1, print_interval=10,
            sample_interval=100)
        gan.save(os.path.join('..', 'models', 'GANterfactual'))

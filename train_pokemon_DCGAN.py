import pickle
from os import listdir
from os.path import isfile, join
import random
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)


def build_generator():
    """
    :return: the generator model.
    """
    noise_shape = (100,)
    model = Sequential(name="generator")
    model.add(Dense(8 * 8 * 512, input_shape=noise_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 512)))

    model.add(Conv2DTranspose(256, 5, 2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, 5, 2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, 5, 2, padding='same', activation='tanh'))
    model.add(BatchNormalization(momentum=0.9))

    model.summary()
    noise = Input(shape=noise_shape)
    image = model(noise)
    return Model(noise, image)


def build_discriminator():
    """
    :return: the discriminator model.
    """
    model = Sequential(name="discriminator")

    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    image = Input(shape=img_shape)
    validity = model(image)
    return Model(image, validity)


def train(epochs, batch_size=128, save_interval=50):
    """
    :param epochs:the number of epochs
    :type epochs: int
    :param batch_size:the batch
    :param save_interval:the number frequency which it will save the model and output images
    """
    try:
        generator = tf.keras.models.load_model('models/generator_model.h5')
        discriminator = tf.keras.models.load_model('models/discriminator_model.h5')
        combined = tf.keras.models.load_model('models/combined_model.h5')
        print("loading...")
    except:
        print("creating new model...")
        optimizer = Adam(0.0002, 0.5)
        discriminator = build_discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        generator = build_generator()
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = Input(shape=(100,))  # Our random input to the generator
        img = generator(z)

        discriminator.trainable = False
        valid = discriminator(img)  # Validity check on the generated image

        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    with open("pokemons.pkl", "rb") as read_file:
        X_train = pickle.load(read_file)["pokemons"]
    X_train = np.array(X_train)
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):

        images = []
        idx = np.random.randint(0, len(X_train), half_batch)
        for i in idx:
            images.append(X_train[i])
        images = np.array(images)

        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_images = generator.predict(noise)

        labels = np.ones((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(images, labels)
        d_loss_fake = discriminator.train_on_batch(gen_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise1 = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise1, valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_images(epoch, generator)
            generator.save('models/generator_model.h5')
            discriminator.save('models/discriminator_model.h5')
            combined.save('models/combined_model.h5')


def save_images(epoch, generator):
    """
    generate an pokemon image with the generator and save it in the batches_output_images folder
    :param epoch:the number of epochs
    :type epoch: int
    :param generator:the generator model
    :type generator: Model
    """
    n = 4
    noise = np.random.normal(0, 1, (n * n, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(gen_images[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
    filename = f"batches_output_images/generated_plot_epoch-{epoch + 1}.png"
    plt.savefig(filename)
    plt.close()


def generate_pokemon():
    """
    generate an pokemon image with the generator and save it in the new_predictions folder
    """
    generator_ = load_model('models/generator_model.h5')
    noise = np.random.normal(0, 1, (1, 100))
    gen_image = generator_.predict(noise)
    gen_image = 0.5 * gen_image + 0.5
    plt.axis("off")
    plt.imshow(gen_image[0])
    plt.savefig(f"new_predictions/p{random.randint(0, 999)}.png")
    plt.close()


def process_data():
    """
    converts the images into a pickle file
    """
    only_files = [f for f in listdir(r"data") if isfile(join(r"data", f))]
    new_data = {}
    new_data["pokemons"] = []
    for i in range(len(only_files)):
        print(i)
        new_data["pokemons"].append(cv2.imread(r"data" + "\\" + only_files[i]))
    new_data["pokemons"] = np.array(new_data["pokemons"])
    new_data["pokemons"] = (new_data["pokemons"].astype(np.float32) - 127.5) / 127.5
    new_data["pokemons"] = new_data["pokemons"].tolist()
    with open("pokemons.pkl", "wb") as write_file:
        pickle.dump(new_data, write_file)


# train(epochs=100000, batch_size=128, save_interval=500)
generate_pokemon()

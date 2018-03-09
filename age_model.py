import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, PReLU, Dropout
from keras.optimizers import Adam

np.random.seed(2 ** 10)


def age_baseline_net(input_shape=None):
    """Ths is Gary's small neural network, running on raspberry pi 3
    """

    inputs = Input(shape=input_shape)

    x = Conv2D(20, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(20, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(20, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)

    pred_age = Dense(1, name='pred_age')(x)
    model = Model(inputs=inputs, outputs=pred_age)

    return model

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, PReLU, Dropout
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet

np.random.seed(2 ** 10)


def age_gender_mobile_net(input_shape=None, alpha=0.1):
    """Ths is Gary's MobileNet, running on smart phone
    """

    inputs = Input(shape=input_shape)
    model_mobile_net = MobileNet(input_shape=input_shape, alpha=alpha, depth_multiplier=1, dropout=1e-3,
                                 include_top=False, weights=None, input_tensor=None, pooling=None)

    x = model_mobile_net(inputs)
    x = Conv2D(20, (1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    pred_age = Dense(1, name='pred_age')(x)
    pred_gender = Dense(1, name='pred_gender')(x)
    model = Model(inputs=inputs, outputs=[pred_age, pred_gender])

    return model


def age_gender_baseline_net(input_shape=None):
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
    pred_gender = Dense(1, name='pred_gender')(x)
    model = Model(inputs=inputs, outputs=[pred_age, pred_gender])

    return model

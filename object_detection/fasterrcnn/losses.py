from keras.models import Model
import keras.layers as KL
import keras.backend as K
from keras.utils.vis_utils import plot_model
import numpy as np


def custom_loss1(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def custom_loss2(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


input_tensor1 = KL.Input((32, 32, 3))
input_tensor2 = KL.Input((4,))
input_target = KL.Input((2,))

x = KL.BatchNormalization(axis=-1)(input_tensor1)

x = KL.Conv2D(16, (3, 3), padding="same")(x)
x = KL.Activation("relu")(x)
x = KL.MaxPool2D(2)(x)

x = KL.Conv2D(32, (3, 3), padding="same")(x)
x = KL.Activation("relu")(x)
x = KL.MaxPool2D(2)(x)

x = KL.Flatten()(x)
x = KL.Dense(32)(x)
out2 = KL.Dense(2)(x)

y = KL.Dense(32)(input_tensor2)
out1 = KL.Dense(2)(y)

loss1 = KL.Lambda(lambda x:custom_loss1(*x), name='loss1')([out2, out1])
loss2 = KL.Lambda(lambda x:custom_loss2(*x), name='loss2')([input_target, out2])

model = Model([input_tensor1, input_tensor2, input_target], [out1, out2, loss1, loss2])

model.summary()

loss_layer1 = model.get_layer("loss1").output
loss_layer2 = model.get_layer("loss2").output
model.add_loss(loss_layer1)
model.add_loss(loss_layer2)

model.compile(optimizer="sgd", loss=[None, None, None, None])



# plot_model(model, to_file="model.png", show_shapes=True)

def data_gen(num):
    for i in range(num):
        yield [np.random.normal(1, 1, (1, 32, 32, 3)), np.random.normal(1, 1, (1, 4)), np.random.normal(1, 1, (1, 2))], []

dataset = data_gen(10000)

model.fit_generator(dataset, steps_per_epoch=100, epochs=20)



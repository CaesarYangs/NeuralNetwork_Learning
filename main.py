
import tensorflow as tf
import keras as keras
from tensorflow.keras import datasets,layers,optimizers

(x,y),(x_val,y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x,dtype=tf.float32)/255.
y = tf.convert_to_tensor(y,dtype=tf.int32)
y = tf.one_hot(y,depth=10)
print(x.shape,y.shape)
db = tf.data.Dataset.from_tensor_slices((x,y)).batch(200)

# for step,(x,y) in enumerate(db):
#     print(step,x.shape,y,y.shape)


model = keras.Sequential([
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)
])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch, train_dataset=None):
    for step,(x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step %100 ==0:
            print(epoch,step,loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch,db)



if __name__ == '__main__':
    train()
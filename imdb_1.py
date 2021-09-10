import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import layers
from keras import models
from keras import optimizers
import numpy as np

#导入imdb数据
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

# print(train_data[0])
# print(train_labels[0])

#数据变换
word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])
decoded_review = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#数据向量化
def vectorize_sequence(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequences in enumerate(sequences):
        results[i,sequences] = 1.
    return results

#数据向量化
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
#标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 构建网络层
imdb_model = models.Sequential()
imdb_model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
imdb_model.add(layers.Dense(16,activation='relu'))
# imdb_model.add(layers.Dense(8,activation='relu'))
# imdb_model.add(layers.Dense(7,activation='relu'))
# imdb_model.add(layers.Dense(6,activation='relu'))
# imdb_model.add(layers.Dense(5,activation='relu'))
# imdb_model.add(layers.Dense(4,activation='relu'))
# imdb_model.add(layers.Dense(3,activation='relu'))
# imdb_model.add(layers.Dense(2,activation='relu'))
imdb_model.add(layers.Dense(1,activation='sigmoid'))





#编译模型
imdb_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#配置优化器
imdb_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#使用自定义损失和指标
from keras import losses
from keras import metrics
imdb_model.compile(optimizer='rmsprop',loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#训练模型
imdb_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = imdb_model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))


#打印训练结果损失等
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
#
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制精确度
# plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
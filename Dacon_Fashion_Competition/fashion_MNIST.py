import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')

train_df = pd.read_csv('Fashion_MNIST/train.csv')
test_df = pd.read_csv('Fashion_MNIST/test.csv')

train_data = np.array(train_df.iloc[:,1:], dtype = 'float32')
test_data = np.array(test_df.iloc[:,1:], dtype='float32')

x_train = train_data[:,1:]/255
y_train = train_data[:,0]
x_test= test_data/255

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_validate = x_validate.reshape(x_validate.shape[0],28,28,1)

cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation = 'relu',input_shape = (28,28,1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(32,activation = 'relu'),
    Dense(10,activation = 'softmax')
])

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=20,
    verbose=1,
    validation_data=(x_validate,y_validate),
)

y_pred = (cnn_model.predict(x_test) > 0.5).astype("int32")

submission = pd.read_csv('Fashion_MNIST/sample_submission.csv', encoding = 'utf-8')
submission['label'] = y_pred
submission.to_csv('Fashion_MNIST/fashion_submission.csv', index = False)

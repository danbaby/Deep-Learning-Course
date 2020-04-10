import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()
#加载数据集
boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_housing.load_data()
print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

num_train=len(train_x)
num_test=len(test_x)
#数据处理
x_train=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
y_train=train_y
x_test=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))
y_test=test_y

x0_train=np.ones(num_train).reshape(-1,1)
x0_test=np.ones(num_test).reshape(-1,1)

X_train=tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
X_test=tf.cast(tf.concat([x0_test,x_test],axis=1),tf.float32)

Y_train=tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test=tf.constant(y_test.reshape(-1,1),tf.float32)

#设置超参数
learn_rate=float(input("请输入学习率："))
display_step=int(input("请输入迭代数:"))
iter=2000
#设置模型变量初始值
np.random.seed(12)
W=tf.Variable(np.random.randn(14,1),dtype=tf.float32)

#训练模型
mse_train=[]
mse_test=[]
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train=tf.matmul(X_train,W)
        Loss_train=0.5*tf.reduce_mean(tf.square(Y_train-PRED_train))
        PRED_test=tf.matmul(X_test,W)
        Loss_test=0.5*tf.reduce_mean(tf.square(Y_test-PRED_test))
    mse_train.append(Loss_train)
    mse_test.append(Loss_test)
    dL_dW=tape.gradient(Loss_train,W)
    W.assign_sub(learn_rate*dL_dW)
    if i % display_step==0:
        print("i:%i,Train Loss:%f,Test Loss:%f"%(i,Loss_train,Loss_test))

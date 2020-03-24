import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

plt.rcParams['font.sans-serif']="SimHei"

for i in range(16):
    num=np.random.randint(1,50000)
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(train_x[num],cmap="gray")
    nums=str(train_y[num])
    plt.title("标签值："+nums,fontsize=14)

plt.tight_layout(rect=[0,0,1,0.9])
plt.suptitle("MNIST测试样本",color="red",fontsize=20)

plt.show()
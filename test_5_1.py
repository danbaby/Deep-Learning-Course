import tensorflow as tf
import numpy as np

x1=tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21]) #面积
x2_=tf.constant([3,2,3,4,1,2,3,2,2,3,1,1,1,1,2,2]) #房间数
y=tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30]) #销售价格

x2=tf.cast(x2_,tf.float32)
x0=tf.ones(16)
X=tf.stack((x0,x1,x2),axis=1)
Y=tf.reshape(y,(-1,1))

Xt=tf.transpose(X) #计算X'
XtX_1=tf.linalg.inv(tf.matmul(Xt,X)) #计算(X'X)-1
XtX_1_Xt=tf.matmul(XtX_1,Xt) #计算(X'X)-1X'
W=tf.matmul(XtX_1_Xt,Y) #计算(X'X)-1X'Y
W=tf.reshape(W,[-1])

sess=tf.Session()
sess.run(tf.global_variables_initializer())
arr=W.eval(session=sess) #将其转换成数组

print("方程为：")
print("Y=",arr[1],"*x1+",arr[2],"*x2+",arr[0])
x=int(input("请输入面积(20-501):"))
while  x<20 or x>500:
    x=int(input("请输入面积(20-501):"))

y=int(input("请输入房间数(1-11):"))
while  y<1 or y>10:
    y=int(input("请输入房间数(1-11):"))
y_pred=arr[1]*x+arr[2]*y+arr[0]
print("预测价格：",round(y_pred,2),"万元")







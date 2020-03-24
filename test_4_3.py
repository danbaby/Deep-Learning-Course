import tensorflow as tf

x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
x_avg=tf.reduce_mean(x)  #平均x
y_avg=tf.reduce_mean(y)  #平均y

xi=tf.subtract(x,x_avg)
yi=tf.subtract(y,y_avg)
wi=tf.multiply(xi,yi)
wi1=tf.reduce_sum(wi)  #w的分子
wj=tf.pow(tf.subtract(x,x_avg),2)
wj1=tf.reduce_sum(wj)  #w的分母

w=tf.divide(wi1,wj1)  #w
sess=tf.compat.v1.Session()
W=w.eval(session=sess)
print("w的值为："+str(W))  #输出w

wx=tf.multiply(w,x_avg)
b=tf.subtract(y_avg,wx)  #b
sess=tf.compat.v1.Session()
B=b.eval(session=sess)
print("b的值为："+str(B))  #输出b

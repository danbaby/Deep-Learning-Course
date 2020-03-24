import tensorflow as tf

x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
n=10
xyi=tf.multiply(x,y)
xys=tf.reduce_sum(xyi)
xs=tf.reduce_sum(x)
ys=tf.reduce_sum(y)
xys1=tf.multiply(xs,ys)
w1=tf.subtract(n*xys,xys1)  #w的分子

xi=tf.pow(x,2)
xis=tf.reduce_sum(xi)
xj=tf.reduce_sum(x)
xjs=tf.pow(xj,2)
w2=tf.subtract(n*xis,xjs)  #w的分母

w=tf.divide(w1,w2)  #w
sess=tf.compat.v1.Session()
W=w.eval(session=sess)
print("w的值为："+str(W))  #输出w

yi=tf.reduce_sum(y)
w3=tf.subtract(yi,w*xs)  #b的分子
b=tf.divide(w3,n)
sess=tf.compat.v1.Session()
B=b.eval(session=sess)
print("b的值为："+str(B))  #输出w
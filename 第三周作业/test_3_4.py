import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(_,_)=boston_housing.load_data(test_split=0)
str="1 -- CRIM\n 2 -- ZN\n3 -- INDUS\n4 -- CHAS\n5 -- NOX\n6 -- RM\n7 -- AGE\n9 -- DIS\n9 -- RAD\n10 -- TAX\n11 --PTRATIO\n12 -- B\n13 -- LSTAT\n14 -- MEDV"
print(str)
num=int(input("请选择属性："))
if num==1:
    plt.figure(figsize=(1,1))#绘图尺寸
    plt.scatter(train_x[:,1],train_y)#绘制散点图
    plt.xlabel("CRIM")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("1. CRIM-PRICE")#设置标题
    plt.show()#显示绘图
elif num==2:
    plt.figure(figsize=(2,2))#绘图尺寸
    plt.scatter(train_x[:,2],train_y)#绘制散点图
    plt.xlabel("ZN")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("2. ZN-PRICE")#设置标题
    plt.show()#显示绘图
elif num==3:
    plt.figure(figsize=(3,3))#绘图尺寸
    plt.scatter(train_x[:,3],train_y)#绘制散点图
    plt.xlabel("INDUS")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("3. INDUS-PRICE")#设置标题
    plt.show()#显示绘图
elif num==4:
    plt.figure(figsize=(4,4))#绘图尺寸
    plt.scatter(train_x[:,4],train_y)#绘制散点图
    plt.xlabel("CHAS")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("4. CHAS-PRICE")#设置标题
    plt.show()#显示绘图
elif num==5:
    plt.figure(figsize=(5,5))#绘图尺寸
    plt.scatter(train_x[:,5],train_y)#绘制散点图
    plt.xlabel("NOX")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("5. NOX-PRICE")#设置标题
    plt.show()#显示绘图
elif num==6:
    plt.figure(figsize=(6,6))#绘图尺寸
    plt.scatter(train_x[:,6],train_y)#绘制散点图
    plt.xlabel("RM")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("6. RM-PRICE")#设置标题
    plt.show()#显示绘图
elif num==7:
    plt.figure(figsize=(7,7))#绘图尺寸
    plt.scatter(train_x[:,7],train_y)#绘制散点图
    plt.xlabel("AGE")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("7. AGE-PRICE")#设置标题
    plt.show()#显示绘图
elif num==8:
    plt.figure(figsize=(8,8))#绘图尺寸
    plt.scatter(train_x[:,8],train_y)#绘制散点图
    plt.xlabel("DIS")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("8. DIS-PRICE")#设置标题
    plt.show()#显示绘图
elif num==9:
    plt.figure(figsize=(9,9))#绘图尺寸
    plt.scatter(train_x[:,9],train_y)#绘制散点图
    plt.xlabel("RAD")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("9. RAD-PRICE")#设置标题
    plt.show()#显示绘图
elif num==10:
    plt.figure(figsize=(10,10))#绘图尺寸
    plt.scatter(train_x[:,10],train_y)#绘制散点图
    plt.xlabel("TAX")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("10. TAX-PRICE")#设置标题
    plt.show()#显示绘图
elif num==11:
    plt.figure(figsize=(11,11))#绘图尺寸
    plt.scatter(train_x[:,11],train_y)#绘制散点图
    plt.xlabel("PTRATIO")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("11. PTRATIO-PRICE")#设置标题
    plt.show()#显示绘图
elif num==12:
    plt.figure(figsize=(12,12))#绘图尺寸
    plt.scatter(train_x[:,12],train_y)#绘制散点图
    plt.xlabel("B")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("12. B-PRICE")#设置标题
    plt.show()#显示绘图
elif num==13:
    plt.figure(figsize=(13,13))#绘图尺寸
    plt.scatter(train_x[:,13],train_y)#绘制散点图
    plt.xlabel("LSTAT")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("13. LSTAT-PRICE")#设置标题
    plt.show()#显示绘图
elif num==14:
    plt.figure(figsize=(14,14))#绘图尺寸
    plt.scatter(train_x[:,14],train_y)#绘制散点图
    plt.xlabel("MEDV")#设置X轴标签
    plt.ylabel("Price($1000's)")#设置y轴标签
    plt.title("14. MEDV-PRICE")#设置标题
    plt.show()#显示绘图
else:
    print("输入不合法！")



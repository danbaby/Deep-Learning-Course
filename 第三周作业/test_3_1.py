import numpy as np
num1=int(input("请输入一个1-100之间的整数："))
np.random.seed(612)
count=1
print("序号\t索引值\t随机数")
for i in range(1,1001):
    num2=np.random.rand()
    if i%num1==0:
        print("%d\t%d\t%.15f"%(count,i,num2))
        count+=1


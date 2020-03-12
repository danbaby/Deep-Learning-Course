import math
a=int(input("请输入a的值："))
b=int(input("请输入b的值："))
c=int(input("请输入c的值："))
num=b*b-4*a*c
if num>0:
    print("经检验，次方程有两个根，分别是：")
    x1=(float)(-b+math.sqrt(num))/(2*a)
    x2=(float)(-b-math.sqrt(num))/(2*a)
    print("x1=%.2f,x2=%.2f"%(x1,x2))
elif num==0:
    print("经检验，此方程有1个根，是")
    x=(float)(-b/(2*a))
    print("x=%.2f"%x)
else:
    print("对不起，该方程无解哦！！！")

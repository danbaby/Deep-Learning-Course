import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif']="SimHei"

img=Image.open("lena.tiff")
img_r,img_g,img_b=img.split()
img1=img_r.resize((50,50))
img21=img_g.transpose(Image.FLIP_LEFT_RIGHT)
img2=img21.transpose(Image.ROTATE_270)
img3=img_b.crop((0,0,150,150))
img4=Image.merge("RGB",[img_r,img_g,img_b])
img4.save("test.PNG")

plt.subplot(221)
plt.axis("off")
plt.imshow(img1,cmap="gray")
plt.title("R-缩放",fontsize=14)

plt.subplot(222)
plt.imshow(img2,cmap="gray")
plt.title("G-镜像+旋转",fontsize=14)

plt.subplot(223)
plt.axis("off")
plt.imshow(img3,cmap="gray")
plt.title("B-裁剪",fontsize=14)

plt.subplot(224)
plt.axis("off")
plt.imshow(img4)
plt.title("RGB",fontsize=14)

plt.suptitle("图像基本操作",fontsize=20,color="blue")
plt.tight_layout(rect=[0,0,1,0.9])

plt.show()

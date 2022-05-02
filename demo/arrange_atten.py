# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import os





methods = ["deit_base_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224"]
images = ["demo1", "demo2", "demo3", "demo4", "demo5", "demo6"]
queries = [23, 83, 98, 143]
heads = [0, 3, 6, 9]
layers = [0, 6, 11]
image_dir = "../images/out/"

figure, axis = plt.subplots((len(layers)+1)*(len(queries))-1, len(heads)+1, figsize=(10,30)) # rows and columns
for i in range((len(layers)+1)*(len(queries))-1):
    for j in range(len(heads)+1):
        axis[i,j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        axis[i,j].spines['top'].set_visible(False)
        axis[i,j].spines['right'].set_visible(False)
        axis[i,j].spines['bottom'].set_visible(False)
        axis[i,j].spines['left'].set_visible(False)
method=methods[0]
image = images[0]
for id_query, query in enumerate(queries):
    img_path = os.path.join(image_dir,method, f"{image}_Q{query}.png")
    axis[id_query*(len(layers)+1)+1, 0].imshow(plt.imread(img_path))
    print(id_query)
    print(img_path)




# # Get the angles from 0 to 2 pie (360 degree) in narray object
# X = np.arange(0, math.pi*2, 0.05)
#
# # Using built-in trigonometric function we can directly plot
# # the given cosine wave for the given angles
# Y1 = np.sin(X)
# Y2 = np.cos(X)
# Y3 = np.tan(X)
# Y4 = np.tanh(X)
# # For Sine Function
# axis[0, 0].imshow(plt.imread("../images/out/vit_base_patch16_224/demo1_Q23_L0_H0.png"))
# axis[0, 0].set_title("Sine Function")
#
# # For Cosine Function
# axis[3, 1].plot(X, Y2)
# axis[3, 1].set_title("Cosine Function")
#
# # For Tangent Function
# axis[1, 0].plot(X, Y3)
# axis[1, 0].set_title("Tangent Function")
#
# # For Tanh Function
# axis[1, 1].plot(X, Y4)
# axis[1, 1].set_title("Tanh Function")

# Combine all the operations and display
plt.show()
# plt.savefig("att.png")

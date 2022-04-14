import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

basemodel = {
    "param": [],
    "mac:": [],
    "acc": [],
    "speed": [],
    "mode": [],
    "meta": "publish info"
}
ResNet = {
    "param": [12, 22, 26, 45, 60],
    "mac:": [1.8, 3.7, 4.1, 7.9, 11.6],
    "acc": [70.6, 75.5, 79.8, 81.3, 81.8],
    "speed": [],
    "mode": ["18", "34", "50", "101", "152"],
    "marker": "v",
    "color": "orange",
    "meta": "arXiv 1 Oct 2021"
}
DeiT = {
    "param": [5, 22, 86],
    "mac:": [1.3, 4.6, 17.6],
    "acc": [72.2, 79.8, 81.8],
    "speed": [],
    "mode": ["T", "S", "B"],
    "marker": "o",
    "color": "C2",
    "meta": "ICML 2021"
}
SwinT = {
    "param": [29, 50, 88],
    "mac:": [4.5, 8.7, 15.4],
    "acc": [81.3, 83, 83.5],
    "speed": [],
    "mode": ["T", "S", "B"],
    "marker": "^",
    "color": "C3",
    "meta": "ICCV 2021"
}
PVT = {
    "param": [13, 25, 44, 61],
    "mac:": [1.9, 3.8, 6.7, 9.8],
    "acc": [75.1, 79.8, 81.2, 81.7],
    "speed": [],
    "mode": ["T","S","M","L"],
    "marker": "<",
    "color": "C4",
    "meta": "ICCV 2021"
}
T2T	= {
    "param": [21.5,39.2,64.1],
    "mac:": [4.8,8.5,13.8],
    "acc": [81.5,81.9,82.3],
    "speed": [],
    "mode": ["14","19","24"],
    "marker": ">",
    "color": "C5",
    "meta": "ICCV 2021"
}
ASMLP = {
    "param": [28, 50, 88],
    "mac:": [4.4, 8.5, 15.2],
    "acc": [81.3, 83.1, 83.3],
    "speed": [],
    "mode": ["T","S","B"],
    "marker": "8",
    "color": "orangered",
    "meta": "ICLR 2022"
}
ConvNeXt = {
    "param": [28, 50, 89, 198],
    "mac:": [4.5, 8.7, 15.4, 45],
    "acc": [82.1, 83.1, 83.8, 84.3],
    "speed": [],
    "mode": ["T","S","B", "L"],
    "marker": "s",
    "color": "violet",
    "meta": "CVPR 2022"
}
PoolFormer = {
    "param": [12,21,31,56],
    "mac:": [2,3.6,5.2,9.1],
    "acc": [77.2,80.3,81.4,82.1],
    "speed": [],
    "mode": ["S12", "S24", "S36", "M36"],
    "marker": "D",
    "color": "deepskyblue",
    "meta": "publish info"
}
ResMLP= {
    "param": [15, 30, 116],
    "mac:": [3, 6, 23],
    "acc": [76.6, 79.4, 81.0],
    "speed": [],
    "mode": ["S12", "S24", "B24"],
    "marker": "*",
    "color": "gold",
    "meta": "arXiv 2021"
}

def var_name(var, all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

save=False
font_size = 10
font_style=""
marker_size=8
line_width=1.5
line_style="-."   # -.   --   -
models = [ResNet, DeiT, SwinT, PVT, T2T, ASMLP, ConvNeXt, PoolFormer, ResMLP]

def plot_param_acc(models = [ResNet, DeiT, SwinT, PVT, T2T, ConvNeXt, PoolFormer, ResMLP]):
    fig, ax = plt.subplots()
    legend_list= []
    for model in models:
        legend_list.append(var_name(model))
        ax.plot(model["param"], model["acc"], marker = model["marker"], markersize = marker_size,
                color=model["color"], linestyle=line_style, linewidth=line_width)
        # ax.scatter(model["param"], model["acc"], marker = model["marker"], s = marker_size,  color =model["color"])
    if ConvNeXt in models:
        ax.set(xlim=(0, 125))
    ax.legend(legend_list)
    # ax.set(xlim=(0, 125), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_param_acc()


#
#
# N = 100
# r0 = 0.6
# x = 0.9 * np.random.rand(N)
# y = 0.9 * np.random.rand(N)
# area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
# c = np.sqrt(area)
# r = np.sqrt(x ** 2 + y ** 2)
# area1 = np.ma.masked_where(r < r0, area)
# area2 = np.ma.masked_where(r >= r0, area)
# plt.scatter(x, y, s=area1, marker='^', c=c)
# plt.scatter(x, y, s=area2, marker='o', c=c)
# # Show the boundary between the regions:
# theta = np.arange(0, np.pi / 2, 0.01)
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
#
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta) + 1)
# plt.show()

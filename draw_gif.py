import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
connectivity_dict = [[0, 1, 0], [1, 2, 0], [2, 3, 0],
                              [3, 4, 0], [1, 5, 1], [5, 6, 1],
                              [6, 7, 1], [1, 8, 0], [8, 9, 0], [9, 10, 0], [10, 11, 0], [8, 12, 1], [12, 13, 1], [13, 14, 1]]

def draw2Dpose(pose_2d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    for i in connectivity_dict:
        x, y = [np.array([pose_2d[i[0], j], pose_2d[i[1], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if i[2] else rcolor)



df = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/predict_sensor_only_no.csv')
list = []
for i in range(15):
    list.append(str(i) + 'x')
    list.append(str(i) + 'y')

headers = list

df.columns = headers
data = df.to_numpy()
ps = []
for d in data:
    p = np.array(d)
    ps.append(np.array(p.reshape(-1,2).tolist()))

i = 0
for td in ps:
    fig = plt.figure(figsize=(4,8))
    ax = fig.add_subplot()
    plt.xlim([500, 1170])
    plt.ylim([0, 1100])
    ax.invert_xaxis()
    ax.invert_yaxis()
    draw2Dpose(td, ax)
    # plt.show()
    fig.savefig('out_predict_fixed_x_final_final/no_{}.png'.format(i))
    i += 1


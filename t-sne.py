import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import csv
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123



def data_scatter(x, colors):
    # choose a color palette with seaborn
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette('hls', num_classes))

    # create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect = 'equal')
    sc = ax.scatter(x[:,0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('tight')
    txts = []
    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)








        # label = f"({l})"
        # plt.annotate(
        #     label,
        #     (x,y),
        #     textcoords='offset points',
        #     xytext=(0,10),
        #     ha='center'
        # )

    # for i in range(num_classes):
    #
    #     # position of each label at median of data points
    #     xtext, ytext = np.median(x[conditions==i,:] , axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground='w'),
    #         PathEffects.Normal()
    #     ])
    #     txts.append(txt)
    plt.savefig('pca_tsne_12.png')

    return f, ax, sc, txts

def pca_function(x):
    time_start = time.time()

    pca = PCA(n_components=30)
    pca_result = pca.fit_transform(x)

    print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

    pca_df = pd.DataFrame(columns= ['pca1', 'pca2', 'pca3', 'pca4'])

    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]

    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

def tsne_function(x):
    pca_30 = PCA(n_components=30)
    pca_result_30 = pca_30.fit_transform(x)
    time_start = time.time()
    fashion_tsne = TSNE(random_state=RS).fit_transform(pca_result_30)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    return fashion_tsne

if __name__ == '__main__':
    feature = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/processed_classification_data/train_feature.csv')
    # x0 = np.load('trainX0.npy')
    # x1 = np.load('trainX1.npy')
    # x2 = np.load('trainX2.npy')
    # X = np.concatenate([x0,x1,x2])
    X = feature.iloc[:,2:].to_numpy()
    # y0 = np.load('name0.npy')
    # y1 = np.load('name1.npy')
    # y2 = np.load('name2.npy')
    # Y = np.concatenate([y0,y1,y2])
    label = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/processed_classification_data/train_lables.csv')
    y_label = label.to_numpy().reshape(-1)

    # w = csv.writer(open('name.csv','w'))
    # for key, value in zip(Y,y_label):
    #     w.writerow([key,value])

    tsne_result = tsne_function(X)
    tsne_df = pd.DataFrame(tsne_result)
    tsne_df.to_csv('tsne.csv')

    f, ax, sc, txts = data_scatter(tsne_result,y_label)
    # df = pd.read_csv('name.csv', header=None)
    # x_column = []
    # y_colunm = []
    # for i in range(150):
    #     x_column.append(condition_xy[i][0])
    # for k in range(150):
    #     y_colunm.append(condition_xy[i][1])
    # df['x_coor'] = pd.Series(np.array(x_column), index=df.index)
    # df['y_coor'] = pd.Series(np.array(y_colunm), index=df.index)
    # df.columns = ['condition', 'number', 'x_coor', 'y_coor']
    # df.to_csv('tsne_data_coordinates.csv')


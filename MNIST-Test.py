from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import kmapper as km
from sklearn.preprocessing import MinMaxScaler

mndata = MNIST('./data/')
mndata.gz = True
images, labels = mndata.load_training()

indexed_images = dict()
for i in range(10):
    indexed_images[i] = []
for i in range(len(images)):
    indexed_images[labels[i]].append(images[i])

def plot(a):
    plt.matshow(np.array(a).reshape((28,28)))
    plt.show()

# plot(indexed_images[0][0])
# from sklearn.preprocessing import MinMaxScaler

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Take only the first 1000 images
# data_im = np.array(images[:1000])
data_im = np.array(images[:])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_im = pca.fit_transform(data_im)

scaler = MinMaxScaler(feature_range=(-1,1))
data_im = scaler.fit_transform(data_im)

# Scatter plot of images PCA'd into R2
colours = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey', 'purple', 'black', 'orange']

# point_colors = [colours[i] for i in labels]

label_indices = dict()
for i in range(10):
    label_indices[i] = []
for i in range(len(labels)):
    label_indices[labels[i]].append(i)

# Bin images by label
data_im_0 = np.array([data_im[i] for i in label_indices[0]])
data_im_1 = np.array([data_im[i] for i in label_indices[1]])
data_im_2 = np.array([data_im[i] for i in label_indices[2]])
data_im_3 = np.array([data_im[i] for i in label_indices[3]])
data_im_4 = np.array([data_im[i] for i in label_indices[4]])
data_im_5 = np.array([data_im[i] for i in label_indices[5]])
data_im_6 = np.array([data_im[i] for i in label_indices[6]])
data_im_7 = np.array([data_im[i] for i in label_indices[7]])
data_im_8 = np.array([data_im[i] for i in label_indices[8]])
data_im_9 = np.array([data_im[i] for i in label_indices[9]])

# Scatter plot by digit
p0 = plt.scatter(data_im_0.T[0].T, data_im_0.T[1].T, c=colours[0])
p1 = plt.scatter(data_im_1.T[0].T, data_im_1.T[1].T, c=colours[1])
p2 = plt.scatter(data_im_2.T[0].T, data_im_2.T[1].T, c=colours[2])
p3 = plt.scatter(data_im_3.T[0].T, data_im_3.T[1].T, c=colours[3])
p4 = plt.scatter(data_im_4.T[0].T, data_im_4.T[1].T, c=colours[4])
p5 = plt.scatter(data_im_5.T[0].T, data_im_5.T[1].T, c=colours[5])
p6 = plt.scatter(data_im_6.T[0].T, data_im_6.T[1].T, c=colours[6])
p7 = plt.scatter(data_im_7.T[0].T, data_im_7.T[1].T, c=colours[7])
p8 = plt.scatter(data_im_8.T[0].T, data_im_8.T[1].T, c=colours[8])
p9 = plt.scatter(data_im_9.T[0].T, data_im_9.T[1].T, c=colours[9])

# Set the legend (colour labels)
plt.legend((p0,p1,p2,p3,p4,p5,p6,p7,p8,p9), ('0','1','2','3','4','5','6','7','8','9'))

# Show the plot
plt.show()

# Fit to and transform the data
projected_data = mapper.fit_transform(data_im, projection='l2norm')

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data_im, nr_cubes=5, overlap_perc=0.1)

# Visualize it
mapper.visualize(graph, path_html="MNIST_PCA_keplermapper_output.html",
                 title="MNIST_PCA_keplermapper")

# Function to print the distribution of labels given a single node
# Assumes a is a list of lists, with each inner list containing a single label
def print_digit_distribution(a):
    bins = dict()
    for i in range(10):
        bins[i] = 0
    for ls in a:
        bins[ls[0]] += 1
    for i in range(10):
        print('{}: {}'.format(i, bins[i]))

# Put image labels into array
labels_array = np.array(labels)
labels_array = labels_array.reshape((60000,1))

# Print distribution of labels for all nodes
for node in graph['nodes']:
    print_digit_distribution(mapper.data_from_cluster_id(node, graph, labels_array))
    print()


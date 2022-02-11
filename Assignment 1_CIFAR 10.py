
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt
import numpy as np

train_dataset = datasets.CIFAR10(root='data/', download=True, train=True, transform=ToTensor())
test_dataset = datasets.CIFAR10(root='data/', download=True, train=False, transform=ToTensor())

np.set_printoptions(threshold=np.inf)
X_train = train_dataset.data
y_train = np.array(train_dataset.targets)
X_test = test_dataset.data
y_test = np.array(test_dataset.targets)
X_test_reshaped = X_test.reshape((10000, 32*32*3))
classes = train_dataset.classes

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(classes[label])
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
plt.show()

import cv2
from google.colab.patches import cv2_imshow # in Google Colab, use cv2_imshow() to show the cv images; use cv2.imshow() to show images if not using Google Colab

print("shape of original image:", X_train[0].shape)
# convert numpy array to cv2 image
img_colored = cv2.cvtColor(X_train[0], cv2.COLOR_BGR2RGB) #colored image
img = cv2.cvtColor(X_train[0], cv2.COLOR_BGR2GRAY) #grayscale image
img_array = np.asarray(img)
print("shape of grayscale image:", img_array.shape)

# display resized image
img_resized = cv2.resize(img, (300, 300))
cv2_imshow(img_resized)
print("shape of resized image:", np.asarray(img_resized).shape)

# apply edge detection
edges = cv2.Canny(img, 200, 200)
# display resized edges
cv2_imshow(cv2.resize(edges, (300, 300)))

# convert back to numpy array
edges_array = np.asarray(edges)
print("shape of edges: ", edges_array.shape)

import numpy as np
import matplotlib.pyplot as plt

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

import copy
import warnings
warnings.filterwarnings("ignore")
class KMeans:
    def __init__(self,number_of_clusters=10,max_iterations=500):
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations
        self.loss_per_iteration = []

    def initiate_centroids(self):
        np.random.seed(42)
        self.centroids = []
        i=0
        while(i<self.number_of_clusters):
            r_index = np.random.choice(range(len(self.data)))
            self.centroids.append(self.data[r_index])
            i=i+1
    
    def init_clusters(self):
        clus = {}
        for i in range(self.number_of_clusters):
          clus[i] = []
        self.clusters = {'data':clus}
        self.clusters['labels']={j:[] for j in range(self.number_of_clusters)}

    def fit(self,data,labels):
        self.data = data
        self.labels = labels
        self.predicted_labels = [None for _ in range(self.data.shape[0])]
        self.initiate_centroids()
        self.iterations = 0
        old_centroids = [np.zeros(shape=(data.shape[1],)) for _ in range(self.number_of_clusters)]
        while not self.converged(self.iterations,old_centroids,self.centroids):
            old_centroids = copy.deepcopy(self.centroids)
            self.init_clusters()
            for j,sample in enumerate(self.data):
                min_dist = float('inf')
                for i,centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample-centroid)
                    if dist<min_dist:
                        min_dist = dist
                        self.predicted_labels[j] = i
                if self.predicted_labels[j] is not None:
                        self.clusters['data'][self.predicted_labels[j]].append(sample)                    
                        self.clusters['labels'][self.predicted_labels[j]].append(self.labels[j])
            self.reshape_cluster()
            self.update_centroids()
            self.calculate_loss()
            print("Iteration:",format(round(self.iterations,2)),'| Loss:',
                  format(round(self.loss,4),".4f"),'| Difference:',
                  format(round(self.centroids_dist,4),".4f"))
            self.iterations+=1
        self.calculate_accuracy()

    def update_centroids(self):
        for i in range(self.number_of_clusters):
            cluster = self.clusters['data'][i]
            if cluster == []:
                self.centroids[i] = self.data[np.random.choice(range(len(self.data)))]
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i],cluster)),axis=0)
    
    def reshape_cluster(self):
        for id,mat in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(mat)

    def converged(self,iterations,centroids,updated_centroids):
        if (iterations > self.max_iterations):
            return True
        self.centroids_dist = np.linalg.norm(np.array(updated_centroids)-
                                             np.array(centroids))
        if self.centroids_dist<=1e-10:
            print("Converged! With distance:",self.centroids_dist)
            return True
        return False

    def calculate_loss(self):
        self.loss = 0
        for key,value in list(self.clusters['data'].items()):
            if value is not None:
                for v in value:
                    self.loss += np.linalg.norm(v-self.centroids[key])
        self.loss_per_iteration.append(self.loss)
    
    def calculate_accuracy(self):
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []
        for clust,labels in list(self.clusters['labels'].items()):
            if isinstance(labels[0],(np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            max_label = max(set(labels), key=labels.count)
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:
                    occur+=1
            acc = occur/len(list(labels))
            self.clusters_info.append([max_label,occur,len(list(labels)),acc])
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy)/self.number_of_clusters
        self.labels_ = []
        for i in range(len(self.predicted_labels)):
            self.labels_.append(self.clusters_labels[self.predicted_labels[i]])

kmeans = KMeans(number_of_clusters=10,max_iterations=1000)
kmeans.fit(X_train, y_train)

print('Accuracy:',kmeans.accuracy)
plt.plot(range(kmeans.iterations),kmeans.loss_per_iteration)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

X_train = train_dataset.data
y_train = np.array(train_dataset.targets)
X_test = test_dataset.data
y_test = np.array(test_dataset.targets)
X_test_reshaped = X_test.reshape((10000, 32*32*3))
classes = train_dataset.classes

# Memory error prevention by subsampling data

num_training = 10000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 1000
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

class KNearestNeighbor():

    def __init__(self):
        pass
    #Passing the train data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    #Predicting the Labels
    def predict(self, X, k=1):
        if num_loops == 0:
            dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)
    #Computing the no loop distance
    def compute_distances(self, X):
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]
      x2 = np.sum(X**2, axis=1).reshape((num_test, 1))
      y2 = np.sum(self.X_train**2, axis=1).reshape((1, num_train))
      xy = -2*np.matmul(X, self.X_train.T)
      dists = np.sqrt(x2 + xy + y2)
      return dists
    
    def predict_labels(self, distances, k=1):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            nearest_y = []
            sorted_dist = np.argsort(distances[i])
            nearest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i]= (np.argmax(np.bincount(nearest_y)))
            pass
        return y_pred

model = KNearestNeighbor()
model.train(X_train, y_train)
dists = model.compute_distances(X_test)
y_pred = model.predict_labels(dists, k=5)
print('%d / %d correct => Accuracy: %f' % (np.sum(y_pred == y_test), num_test, (np.sum(y_pred == y_test))/num_test))

num_folds = 5
k_values = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_trainf = []
y_trainf = []

X_trainf = np.array_split(X_train, num_folds)
y_trainf = np.array_split(y_train, num_folds)
k_to_accuracies={}
for k in k_values:
    k_to_accuracies[k] = []

for k in k_values:
    for i in range(num_folds):
        X_traincv = np.vstack(X_trainf[0:i] + X_trainf[i+1:])
        X_test_crossval = X_trainf[j]

        y_traincv = np.hstack(y_trainf[0:i]+y_trainf[i+1:])
        y_test_crossval = y_trainf[j]
        model.train(X_traincv, y_traincv)
        dists_crossval = model.compute_distances(X_test_crossval)
        y_test_pred = model.predict_labels(dists_crossval, k)
        k_to_accuracies[k].append(float(np.sum(y_test_pred == y_test_crossval)) / num_test)

for k in k_values:
    accuracies = k_to_accuracies[k]
    print((k, accuracies))
    plt.scatter([k] * len(accuracies), accuracies)
a_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
a_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_values, a_mean, yerr=a_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

X_train = train_dataset.data
y_train = np.array(train_dataset.targets)
X_test = test_dataset.data
y_test = np.array(test_dataset.targets)
X_test_reshaped = X_test.reshape((10000, 32*32*3))
classes = train_dataset.classes

from random import randrange


def eval_numerical_gradient(f, x):

    fx = f(x)  # Evaluate f(x)
    grad = np.zeros(x.shape)  # Initializaion of the gradient
    h = 1e-5  # Increment size

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluate function at x+h
        ix = it.multi_index
        x[ix] += h  # Increment by h at dimension index ix
        fxh = f(x)  # Evaluate f(x + h)
        x[ix] -= h  # Restore to previous value, or the partial derivative and f(x+h) in the next step will be affected

        # Compute the partial derivative
        grad[ix] = (fxh - fx) / h  # Calculate the slope
        print(ix, grad[ix])
        it.iternext()  # Step to the next dimension index

    return grad


def grad_check(f, x, analytic_grad, num_check_pairs):
    h = 1e-5

    for i in range(num_check_pairs):
        # Here the numerical gradients are calculated using df/dx = (f(x+h)+f(x-h))/2h
        ix = tuple([randrange(m) for m in x.shape])
        x[ix] += h  # Increment by h at dimension index ix
        fxph = f(x)  # Evaluate f(x + h)
        x[ix] -= 2 * h  # Decrement by h at dimension index ix
        fxmh = f(x)  # Evaluate f(x - h)
        x[ix] += h  # Reset x

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        relative_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, relative_error))

class SoftmaxClassifier:
    def __init__(self):
        C, D = 10, 3073
        self.W = np.random.randn(C, D) * 0.001

    def train(self, x, y, lr=1e-5, reg=1e-3, num_iters=1000, batch_size=200):
        N = x.shape[1]
        loss_record = []
        for it in range(num_iters):
            indices = np.random.choice(N, batch_size, replace=True)
            x_batch = x[:, indices]
            y_batch = y[indices]
            loss, grad = self.cross_entropy_loss(x_batch, y_batch, reg)
            loss_record.append(loss)
            self.W -= lr * grad

        return loss_record

    def predict(self, x):
        y = self.W.dot(x)
        y_pred = np.argmax(y, axis=0)
        return y_pred

    def softmax(self,z):
      return np.exp(z) / np.sum(np.exp(z), axis=0)

    def cross_entropy_loss(self, x, y, reg):
        z = np.dot(self.W, x)
        z -= np.max(z, axis=0)
        p = self.softmax(z)
        L = -1 / len(y) * np.sum(np.log(p[y, range(len(y))]))
        R = 0.5 * np.sum(np.multiply(self.W, self.W))
        loss = L + R * reg

        # Calculation of dW
        p[y, range(len(y))] -= 1
        dW = 1 / len(y) * p.dot(x.T) + reg * self.W
        return loss, dW

x_train = np.reshape(X_train, (X_train.shape[0], -1))
x_test = np.reshape(X_test, (X_test.shape[0], -1))

# Add a row in data to fit the relation y = Wx.
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]).T
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))]).T

classifier = SoftmaxClassifier()
loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

print("Learning rate: 1e-5, Regularisation: 1e4")
# Gradient check for the model
f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
print('Gradient Check:')
grad_check(f, classifier.W, grad, 10)
print()

    # Plot the loss for the training

loss_record = classifier.train(x_train, y_train, lr=1e-5, reg=1e4)
plt.plot(loss_record)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

    # Evaluation on test set
y_test_pred = classifier.predict(x_test)
accuracy = np.mean(y_test == y_test_pred)
print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

classifier = SoftmaxClassifier()
loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

print("Learning rate: 1e-5, Regularisation: 1e4")
# Gradient check for the model
f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
print('Gradient Check:')
grad_check(f, classifier.W, grad, 10)
print()

    # Plot the loss for the training

loss_record = classifier.train(x_train, y_train, lr=1e-5, reg=1e4)
plt.plot(loss_record)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

    # Evaluation on test set
y_test_pred = classifier.predict(x_test)
accuracy = np.mean(y_test == y_test_pred)
print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

classifier = SoftmaxClassifier()
loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

print("Learning rate: 1e-5, Regularisation: 1e3")
# Gradient check for the model
f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
print('Gradient Check:')
grad_check(f, classifier.W, grad, 10)
print()

    # Plot the loss for the training

loss_record = classifier.train(x_train, y_train, lr=1e-5, reg=1e3)
plt.plot(loss_record)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

    # Evaluation on test set
y_test_pred = classifier.predict(x_test)
accuracy = np.mean(y_test == y_test_pred)
print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

classifier = SoftmaxClassifier()
loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

print("Learning rate: 1e-6, Regularisation: 1e4")
# Gradient check for the model
f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
print('Gradient Check:')
grad_check(f, classifier.W, grad, 10)
print()

    # Plot the loss for the training

loss_record = classifier.train(x_train, y_train, lr=1e-6, reg=1e4)
plt.plot(loss_record)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

    # Evaluation on test set
y_test_pred = classifier.predict(x_test)
accuracy = np.mean(y_test == y_test_pred)
print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

classifier = SoftmaxClassifier()
loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

# Gradient check for the model
f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
print('Gradient Check:')
grad_check(f, classifier.W, grad, 10)
print()

    # Plot the loss for the training
loss_record = classifier.train(x_train, y_train, lr=1e-6, reg=1e3)
plt.plot(loss_record)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

    # Evaluation on test set
y_test_pred = classifier.predict(x_test)
accuracy = np.mean(y_test == y_test_pred)
print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

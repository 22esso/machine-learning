import numpy as np
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def get_non_separable_dataset():
    # To reset the randomization each time the function is called
    # generating 300 samples with 2 features each
    X =np.random.randn(50,2)
    # using the XOR functionality,so I can get different intersected data samples
    Y=np.logical_xor(X[:, 0 ] > 0, X[:, 1] > 0)
    # changing the logical values with integer labels of 0 and 1
    Y = np.where(Y,-1,1)
    # shuffle the data set to ensure the maximum randomization
    X,Y = shuffle_arrays_unison(arrays=[X,Y],random_seed=42)
    return X,Y

def get_separable_dataset():
    X, Y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    Y = np.where(Y == 0, -1, 1)
    X, Y = shuffle_arrays_unison(arrays=[X, Y], random_seed=42)
    return X, Y


def plot_dataset(X,Y):
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X[Y == -1, 0],
                X[Y == -1, 1],
                c='b', marker='x',
                label='-1')
    plt.scatter(X[Y == 1, 0],
                X[Y == 1, 1],
                c='r',
                marker='s',
                label='1')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[Y == -1, 0],
                X[Y == -1, 1],
                c='b', marker='x',
                label='-1')
    plt.scatter(X[Y == 1, 0],
                X[Y == 1, 1],
                c='r',
                marker='s',
                label='1')

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()



if __name__ == "__main__":
   #np.random.seed(0)
   X,Y=get_non_separable_dataset()
   #plot_dataset(X,Y)
   clf = SVM()
   clf.fit(X,Y)
   X_train, X_test, y_train, y_test = train_test_split(
       X, Y, test_size=0.2, random_state=123
   )
   predictions = clf.predict(X_test)
   #visualize_svm()

   print("SVM classification accuracy", accuracy(y_test, predictions))
   print(f"Optimal weights : {clf.w}")
   print(f"Optimal alpha (learning rate) : {clf.lr}")
   print(f"Optimal bias : {clf.b}")


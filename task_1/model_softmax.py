import numpy as np


class softmax_regression():
    def __init__(self):
        self.batch_size = None
        self.epochs = None
        self.lr = None
        self.vector_size = None
        self.w = None
        self.b = None
        self.num_classes = None
        self.acc_score = 0

    def train(self, feature_vectors, labels, lr=0.1, epochs=10, num_classes=5):
        self.__init__()

        self.num_classes = num_classes
        feature_vectors = np.array(feature_vectors)  # 10000*vector_size
        validation_data = np.array(feature_vectors[:1000])
        labels = [int(label) for label in labels]
        labels = np.array(labels)

        self.batch_size, self.vector_size = feature_vectors.shape
        self.w = np.random.randn(self.vector_size, self.num_classes)  # vector_size*5
        self.b = np.zeros(num_classes)
        w = self.w
        b = self.b
        labels_one_hot = np.zeros((self.batch_size, self.num_classes))
        for index in range(self.batch_size):
            labels_one_hot[index][int(labels[index])] += 1  #

        for i in range(epochs):
            feature_vectors_list = np.array_split(feature_vectors, 1000)
            labels_one_hot_lsit = np.array_split(labels_one_hot, 1000)
            for v, l in zip(feature_vectors_list, labels_one_hot_lsit):
                pred = np.dot(v, self.w) + self.b
                pred_labels = softmax(pred)  # 100*5
                loss = -np.mean(np.log(pred_labels) * l)
                grad_scores = pred_labels - l  # 100*5
                grad_w = np.dot(v.T, grad_scores)  # vector_size*5
                grad_b = np.sum(grad_scores, axis=0)
                self.w -= grad_w * lr
                self.b -= grad_b * lr
            new_acc = self.acc(validation_data, labels_one_hot[:1000])
            if self.acc_score < new_acc:
                self.acc_score = new_acc
                w = self.w
                b = self.b
            if (i + 1) % 10 == 0:
                print(f"epcoh {i} loss is :{loss}, best acc is {self.acc_score}, acc is {new_acc}")
        # start gradient descent
        return w, b

    def predict(self, x):
        pred = np.dot(x, self.w)  + self.b
        pred_softmax = softmax(pred)
        prediction = np.argmax(pred_softmax, axis=1)
        return prediction

    def acc(self, validation_data, labels):
        score = 0
        validation = self.predict(validation_data)
        labels = np.argmax(labels, axis=1)
        for validation_item, label in zip(validation, labels):
            if validation_item == label:
                score += 1
        score /= len(labels)
        return score


def cross_entropy_loss(pred_labels, labels):
    num_samples = labels.shape[0]
    loss = -np.sum(np.dot(labels.T, np.log(pred_labels))) / num_samples
    return loss


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    z = z / np.sum(z, axis=1, keepdims=True)
    return z

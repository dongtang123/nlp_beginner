import numpy as np

def fit(self, X, y, learning_rate=0.001, epochs=10, num_classes=5):
    '''

    :param X: [batch_size, num_features]
    :param y: [batch_size, 1]
    :param w: [num_classes, num_features]
    :return:

    '''
    self.__init__(epochs, learning_rate=learning_rate)
    self.batch_size, self.num_features = X.shape
    self.num_classes = num_classes
    self.w = np.random.randn(self.num_classes, self.num_features)

    y_one_hot = np.zeros((self.batch_size, self.num_classes))
    for i in range(self.batch_size):
        y_one_hot[i][y[i]] = 1 #把y所属的类标记为1

    loss_history = []

    for i in range(epochs):
        loss = 0
        probs = X.dot(self.w.T)
        probs = softmax(probs)
        for i in range(self.batch_size):
            loss -= np.log(probs[i][y[i]])
        weight_update = np.zeros_like(self.w)
        for i in range(self.batch_size):
            weight_update += X[i].reshape(1, self.num_features).T.dot((y_one_hot[i] - probs[i]).reshape(1, self.num_classes)).T# 800*1   5*1
            #拿出X的第i行
        self.w += weight_update * self.learning_rate / self.batch_size

        loss /= self.batch_size
        loss_history.append(loss)
        if i % 10 == 0:
            print("epoch {} loss {}".format(i, loss))

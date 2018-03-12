import cv2
import fetch_MNIST
import layer_factory_batch as network
import numpy as np


class CovNet:

    def __init__(self):
        self.max_iteration = 10000
        self.cov1 = network.Cov(1, 6, kernel_size=5)   # shape (  6,  4,  4)
        self.pool1 = network.Pool(2)                   # shape (  6, 14, 14)
        self.cov2 = network.Cov(6, 16, kernel_size=5)   # shape ( 16, 14, 14)
        self.pool2 = network.Pool(2)                   # shape ( 16,  7,  7)
        self.fc1 = network.FullyConect(784, 200)       # shape (200,)
        self.fc2 = network.FullyConect(200, 10)        # shape ( 10,)

    def img_preprocess(self, img):
        img = cv2.resize(img, (28, 28))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.
        return img

    def loss(self, feature):
        """
        error :       L2 loss
        error_delta:  loss backward delta
        """
        # 0.5 * (feature - label)^2
        error = np.array(0.5 * pow(feature - self.label, 2))
        error_delta = feature - self.label
        return error_delta, error

    def run(self, data, label):
        # for i in xrange(self.max_iteration):
        # self.data = self.img_preprocess(data)
        self.data = data / 225.
        self.label = label

        cov1_forward = self.cov1.foward(self.data)
        pool1_forward = self.pool1.foward(cov1_forward)
        cov2_forward = self.cov2.foward(pool1_forward)
        pool2_forward = self.pool2.foward(cov2_forward)
        fc1_forward = self.fc1.foward(pool2_forward.reshape(1, -1))
        fc2_forward = self.fc2.foward(fc1_forward)
        delta, error = self.loss(fc2_forward)
        # backward
        fc2_delta = self.fc2.backward(delta)
        fc1_delta = self.fc1.backward(fc2_delta)
        pool2_delta = self.pool2.backward(fc1_delta.reshape(pool2_forward.shape))
        cov2_delta = self.cov2.backward(pool2_delta)
        pool1_delta = self.pool1.backward(cov2_delta)
        self.cov1.backward(pool1_delta)
        return fc2_forward
        # log
        # print "iteration {:3} pred = {}".format(i,fc2_forward[0 ,0])
        # print "          {:3} loss = {}".format(' ', error)

def convertToOneHot(labels):
    oneHotLabels = np.zeros((labels.size, labels.max()+1))
    oneHotLabels[np.arange(labels.size), labels] = 1
    return oneHotLabels

def shuffle_dataset(data, label):
    N = data.shape[0]
    index = np.random.permutation(N)
    x = data[index, :, :]; y = label[index, :]
    return x, y

if __name__ == '__main__':
    train_imgs = fetch_MNIST.load_train_images()
    train_labs = fetch_MNIST.load_train_labels().astype(int)
    # size of data;                  batch size
    data_size = train_imgs.shape[0]; batch_sz = 1;
    # learning rate; max iteration;    iter % mod (avoid index out of range)
    lr = 0.0001;     max_iter = 50000; iter_mod = int(data_size/batch_sz)
    train_labs = convertToOneHot(train_labs)
    lenet = CovNet()
    correct_list = []
    for iters in range(max_iter):
        st_idx = (iters % iter_mod) * batch_sz
        if st_idx == 0:
            print train_imgs.shape, train_labs.shape
            train_imgs, train_labs = shuffle_dataset(train_imgs, train_labs)
        input_data = train_imgs[st_idx : st_idx + batch_sz]
        output_label = train_labs[st_idx : st_idx + batch_sz]

        softmax_output = lenet.run(input_data, output_label[0])
        correct_list.append(int(np.argmax(softmax_output[0]) == np.argmax(output_label[0])))
        if iters % 50 == 0:
            # calculate accuracy
            print correct_list
            accuracy = float(np.array(correct_list).sum()) / 50
            # calculate loss
            correct_prob = [ softmax_output[i][np.argmax(output_label[i])] for i in range(batch_sz) ]
            correct_prob = filter(lambda x: x > 0, correct_prob)
            loss = -1.0 * np.sum(np.log(correct_prob))
            print "The %d iters result:" % iters
            print "The accuracy is %f The loss is %f " % (accuracy, loss)
            correct_list = []


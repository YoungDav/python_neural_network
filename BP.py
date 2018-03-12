import numpy as np
import layer_factory as network


class BP:
    def __init__(self):
        # self.data = np.random.randint(5, size = (1, 5))
        # self.label = np.random.randint(5, size = (1, 1))
        self.data = np.array([[4, 5]])
        self.label = np.array([[1]])
        self.max_iteration = 1000

        # network data -> fc1 -> fc2 -> label
        self.fc1 = network.FullyConect(2, 2)
        self.fc2 = network.FullyConect(2, 1)

    def loss(self, feature):
        """
        error :       L2 loss
        error_delta:  loss backward delta
        """
        # 0.5 * (feature - label)^2
        error = np.array(0.5 * pow(feature - self.label, 2))
        error_delta = feature - self.label
        return error_delta, error

    def run(self):
        for i in xrange(self.max_iteration):
            # forward
            fc1_forward = self.fc1.foward(self.data)
            fc2_forward = self.fc2.foward(fc1_forward)
            delta, error = self.loss(fc2_forward)
            # backward
            fc2_delta = self.fc2.backward(delta)
            self.fc1.backward(fc2_delta)
            # log
            print "iteration {:3} pred = {}".format(i,fc2_forward[0, 0])
            print "          {:3} loss = {}".format(' ', error[0, 0])
bp = BP()
bp.run()





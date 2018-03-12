import numpy as np

def xavier_init(c1, c2, w=1, h=1, fc=False, cov1d=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2*np.random.random((c1, c2, w, h)) - 1)
    if cov1d:
        params = params.reshape(c1, c2, -1)
    if fc == True:
        params = params.reshape(c1, c2)
    return params

class Cov1d:
    """
    SAME method: input shape = output shape
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1):
        self.channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        # weight: shape N * K * F
        self.weight = xavier_init(self.channel, self.kernel_size, input_channel, cov1d=True)
        # self.weight = np.random.randint(50, size = (self.channel, self.kernel_size, input_channel)) * 1.0
        self.lr = 0.001

    def padding(self, feature, forward=True):
        padding_size = self.kernel_size - 1
        if forward:
            # N * F -> N * pad(F)
            padding_feature = np.pad(feature, ((padding_size / 2, padding_size / 2), (0, 0)), 'constant', constant_values=0)
        else:
            padding_feature = feature[padding_size / 2: -1 * padding_size / 2, :]
        return padding_feature

    def im2col(self, img):
        col_list = []
        for i in xrange(self.N):
            col_sub = img[i:i + self.kernel_size, :].reshape(-1)
            col_list.append(col_sub)
        return np.array(col_list)

    def col2im(self, col):
        img_output = np.zeros([self.N + self.kernel_size - 1, self.F])
        # for n in xrange(0, self.N, self.stride):
        n = 0
        for c in col:
            img_output[n:n + self.kernel_size, :] += c.reshape(self.kernel_size, -1)
            n += 1
        return img_output

    def filter2col(self, filter):
        return filter.reshape(self.channel, -1)

    def col2filter(self, col):
        filter = col.reshape(self.channel, self.kernel_size, self.F)
        return filter

    def forward(self, feature):
        # SAME
        self.feature = feature
        self.N, self.F = feature.shape
        feature_padding = self.padding(self.feature)
        # N * feature_channel -> N * (feature_channel * kernel_sizee)
        feature_col = self.im2col(feature_padding)
        # filters * feature_channel * kernel_size -> filters * (feature_channel * kernel_size)
        filter_col = self.filter2col(self.weight)
        feature_covd = np.dot(feature_col, filter_col.T)
        return feature_covd

    def backward(self, delta):
        feature_padding = self.padding(self.feature)
        feature_col = self.im2col(feature_padding)

        # d(loss) / d(weight)
        delta_weight = np.dot(delta.T, feature_col)
        delta_weight = self.col2filter(delta_weight)

        # d(loss) / d(input)
        delta_feature = np.dot(delta, self.filter2col(self.weight))
        delta_feature = self.col2im(delta_feature)

        # update
        self.weight -= self.lr * delta_weight

        return self.padding(delta_feature, forward=False)

class Cov:
    """
    SAME method: input shape = output shape
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1):
        self.channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        # self. weight = np.random.randint(10, size=(self.channel, input_channel, self.kernel_size, self.kernel_size)) * 1.
        self.weight = xavier_init(self.channel, input_channel, self.kernel_size, self.kernel_size)
        self.lr = 0.001

    def filter2col(self, filter):
        """
        filter:  shape [C, N, kernel_size x kernel_size]
        col:     shape [N, C x kernel_size x kernel_size]
        """
        return filter.reshape(self.channel, -1)

    def col2filter(self, col):
        """
        filter:  shape [C, N, kernel_size x kernel_size]
        col:     shape [N, C x kernel_size x kernel_size]
        """
        return col.reshape(self.channel, self.C, self.kernel_size, self.kernel_size)

    def im2col(self, im):
        """
        * im -> col for forward
        im:      shape [C, H, W]
        col:     shape [H x W, C x kernel_size x kernel_size]
        """
        col_output = []
        for h in xrange(0, self.H, self.stride):
            for w in xrange(0, self.W, self.stride):
                end_h = h + self.kernel_size
                end_w = w + self.kernel_size
                feature_sub = im[:, h:end_h, w:end_w].flatten()
                col_output.append(feature_sub)
        return np.array(col_output)

    def col2im(self, col):
        """
        * col -> im  for backward
        im:      shape [C, H, W]
        col:     shape [H x W, C x kernel_size x kernel_size]
        """
        img_output = np.zeros([self.C, self.H + self.kernel_size - 1, self.W + self.kernel_size - 1])
        for h in xrange(0, self.H, self.stride):
            for w in xrange(0, self.W, self.stride):
                end_h = h + self.kernel_size
                end_w = w + self.kernel_size
                channel_stride = self.kernel_size * self.kernel_size
                for c in xrange(0, self.C):
                    img_output[c, h:end_h, w:end_w] += col[h * self.W + w][c * channel_stride: (c + 1) * channel_stride].reshape(self.kernel_size, self.kernel_size)
        return img_output

    def padding(self, feature, forward=True):
        padding_size = self.kernel_size - 1
        if forward:
            padding_feature = np.pad(feature, ((0, 0), (padding_size / 2, padding_size / 2), (padding_size / 2, padding_size / 2)), 'constant', constant_values=0)
        else:
            padding_feature = feature[:, padding_size / 2: -1 * padding_size / 2, padding_size / 2: -1 * padding_size / 2]
        return padding_feature

    def foward(self, feature):
        # SAME
        self.feature = feature
        self.C, self.H, self.W = feature.shape
        feature_padding = self.padding(self.feature)
        feature_col = self.im2col(feature_padding)
        filter_col = self.filter2col(self.weight)
        feature_covd = np.dot(feature_col, filter_col.T)
        # [H x W, C] -reshape-> [C, H, W]
        return feature_covd.T.reshape(self.channel, self.H, self.W)

    def backward(self, delta):
        feature_padding = self.padding(self.feature)
        feature_col = self.im2col(feature_padding)

        delta_col = self.filter2col(delta).T
        # d(loss) / d(weight)
        delta_weight = np.dot(delta_col.T, feature_col)

        # d(loss) / d(input)
        delta_feature = np.dot(delta_col, self.filter2col(self.weight))
        delta_feature = self.col2im(delta_feature)

        # update
        self.weight -= self.lr * self.col2filter(delta_weight)

        return self.padding(delta_feature, forward=False)

class FullyConect:
    """
    fetaure             shape C
    Weight              shape N x C
    Bias                shape 1
    delta               shape N
    filter              shape N
    channel             shape C

    feature forward     Weight * fetaure

    Weight derivative   fetaure * delta
    Bias derivative     sum(delata)
    """
    def __init__(self, channel, filter):
        self.filter = filter
        self.channel = channel
        self.weight = xavier_init(self.filter, self.channel, fc=True)
        # self.weight = np.random.randint(10, size = (self.filter, self.channel)) * 1.
        self.bias = 2
        self.lr = 0.001

    def foward(self, feature):
        # for Matrix transposed .T
        # feature = np.expand_dims(feature, axis=0)

        self.input = feature
        feature_forward = np.dot(self.weight, feature.T).T + self.bias
        return feature_forward

    def updates(self, delta):
        self.weight_delta = np.dot(delta.T, self.input)
        self.bias_delta = np.sum(delta)
        self.weight -= self.lr * self.weight_delta
        self.bias -= self.lr * self.bias_delta

    def backward(self, delta):
        # for Matrix transposed .T
        # delta = np.expand_dims(delta, axis=0)

        derivative = np.dot(delta, self.weight)
        self.updates(delta)
        return derivative

class Pool:
    def __init__(self, stride=2):
        self.stride = stride
        self.type = type

    def mean_pool(self, feature):
        feature_pooled = np.zeros([self.h_index, self.w_index])
        for h in xrange(self.h_index):
            for w in xrange(self.w_index):
                h_min = max(h * self.stride, 0)
                w_min = max(w * self.stride, 0)
                h_max = min((h + 1) * self.stride, self.H)
                w_max = min((w + 1) * self.stride, self.W)
                feature_pooled[h][w] = np.mean(feature[h_min:h_max, w_min:w_max])
        return feature_pooled

    def foward(self, feature):
        self.C, self.H, self.W = feature.shape
        # feature -split-> block
        self.h_index, self.w_index = int(np.ceil(self.H / float(self.stride))), int(np.ceil(self.W / float(self.stride)))
        feature_pooled = np.zeros([self.C, self.h_index, self.w_index])
        for c in xrange(self.C):
            feature_pooled[c, :] = self.mean_pool(feature[c, :])
        return feature_pooled

    def backward(self, delta):
        # input size = output size
        delta = delta * 1.
        delta = delta.repeat(self.stride, axis=1).repeat(self.stride, axis=2)[:, :self.H, :self.W]
        for h in xrange(self.h_index):
            for w in xrange(self.w_index):
                h_min = max(h * self.stride, 0)
                w_min = max(w * self.stride, 0)
                h_max = min((h + 1) * self.stride, self.H)
                w_max = min((w + 1) * self.stride, self.W)
                element_number = ((h_max - h_min) * (w_max - w_min))
                delta[:, h_min:h_max, w_min:w_max] = delta[:, h_min:h_max, w_min:w_max] / float(element_number)
        return delta

class Relu:
    def __init__(self):
        # where weights < 0 -> 0
        self.zero_index = 0

    def foward(self, feature):
        self.zero_index = np.where(feature < 0)
        feature[self.zero_index] = 0
        return feature

    def backward(self, delta):
        delta[self.zero_index] = 0
        return delta
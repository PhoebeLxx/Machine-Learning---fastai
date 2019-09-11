import pickle
import numpy as np
# from app.model.preprocessor import Preprocessor as img_prep


class LiteOCR:
    def __init__(self, fn='alpha_weights.pkl', pool_size=2):
        [weights, meta] = pickle.load(open(fn, 'rb'), encoding='latin1')
        self.vocab = meta['vocab']

        self.img_rows = meta['img_side']
        self.img_cols = meta['img_side']
        self.CNN = LiteCNN()
        self.CNN.load_weights(weights)
        self.CNN.pool_size = int(pool_size)
        
    def predict(self, image):
        print(image.shape)

        X = np.reshape(image, (1, 1, self.img_rows, self.img_cols))
        X = X.astype('float32')

        predicted_i = self.CNN.predict(X)
        return self.vocab[predicted_i]


class LiteCNN:
    def __init__(self):
        self.layers = []
        self.pool_size = None
    
    def load_weights(self, weights):
        assert not self.layers, "Weights can only be loaded once!"

        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])

    def predict(self, X):
        h = self.cnn_layer(X, layer_i=0, border_mode='full'); X = h
        h = self.relu_layer(X); X = h
        h = self.cnn_layer(X, layer_i=2, border_mode='valid'); X = h
        h = self.relu_layer(X); X = h
        h = self.maxpooling_layer(X); X = h
        h = self.dropout_layer(X, 0.25); X = h
        h = self.flatten_layer(X); X = h
        h = self.dense_layer(X, layer_i=10); X = h
        h = self.softmax_layer2D(X); X = h
        max_i = self.classify(X)
        return max_i[0]
    
    def maxpooling_layer(self, convolved_features):
        nb_features = convolved_features.shape[0]
        nb_images = convolved_features.shape[1]
        conv_dim = convolved_features.shape[2]
        res_dim = int(conv_dim / self.pool_size)

        pooled_features = np.zeros((nb_features, nb_images, res_dim, res_dim))
        for image_i in range(nb_images):
            for feature_i in range(nb_features):
                row_start = pool_row * self.pool_size
                row_end = row_start + self.pool_size

                for pool_row in range(res_dim):
                    row_start = pool_row * self.pool_size
                    row_end = row_start + self.pool_size

                    for pool_col in range(res_dim):
                        col_start = pool_col * self.pool_size
                        col_end = col_start + self.pool_size

                        patch = convolved_features[feature_i, image_i, row_start:row_end, col_start:col_end]
                        pooled_features[feature_i, image_i, pool_row, pool_col] = np.max(patch)

        return pooled_features
    
    def cnn_layer(self, X, layer_i=0, border_mode='full'):
        features = self.layers[layer_i]['param_0']
        bias = self.layers[layer_i]['param_1']

        patch_dim = features[0].shape[-1]
        
        nb_features = features.shape[0]

        image_dim = X.shape[2]
        image_channels = X.shape[1]

        nb_images = X.shape[0]

        if border_mode == 'full':
            conv_dim = image_dim + patch_dim - 1
        elif border_mode == 'valid':
            conv_dim = image_dim - patch_dim + 1
        
        convolved_features = np.zeros((nb_images, nb_features, conv_dim))

        for image_i in range(nb_images):
            for feature_i in range(nb_features):
                convolved_image = np.zeros((conv_dim, conv_dim))
                for channel in range(image_channels):
                    feature = feature[feature_i, channel, :, :]
                    image = X[image_i, channel, :, :]
                    convolved_image += self.convolve2d(image, feature, border_mode)
                
                convolved_image = convolved_image + bias[feature_i]
                convolved_features[image_i, feature_i, :, :] = convolved_image
        
        return convolved_features
    
    def dense_layer(self, X, layer_i=0):
        W = self.layers[layer_i]['param_0']
        b = self.layers[layer_i]['param_1']
        
        output = np.dot(X, W) + b
        return output

    @staticmethod
    def convolve2d(image, feature, border_mode='full'):
        image_dim = np.array(image.shape)
        feature_dim = np.array(feature.shape)

        target_dim = image_dim + feature_dim - 1

        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim) 
        target = np.fft.ifft2(fft_result).real

        if border_mode == 'valid':
            valid_dim = image_dim - feature_dim + 1

            if np.any(valid_dim < 1):
                valid_dim = feature_dim - image_dim + 1
            start_i = (target_dim - valid_dim) // 2
            end_i = start_i + valid_dim
            target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]

        return target

    @staticmethod
    def relu_layer(self, x):
        z = np.zeros_like(x)
        return np.where(x>z, x, z)

    @staticmethod
    def softmax_layer2D(self, w):
        maxes = np.amax(w, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(w - maxes)
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist

    @staticmethod
    def dropout_layer(self, X, p):
        retain_prob = 1. - p
        X *= retain_prob
        return X

    @staticmethod
    def classify(self, X):
        return X.argmax(axis=-1)

    @staticmethod
    def flatten_layer(self, X):
        flatX = np.zeros((X.shape[0], np.prod(X.shape[1:])))
        for i in range(X.shape[0]):
            flatX[i:] = X[i].flatten(order='C')
        return flatX


import keras
import numpy as np
import resnet_utils.resnet_data_utils as ut

from sklearn.utils import shuffle
from data_augmentation import noise, shift, stretch, pitch, dyn_change, speed_npitch,mix_up,spec_augment

def get_augmention_data(self, x, y, ratio=0.2, mix_ratio=0.1, origin=True):
    x_len = len(x)
    indices = range(x_len)
    sample_num = int(x_len * ratio)
    augmention_data_x = []
    augmention_data_y = []
    if origin:
        augmention_data_x.append(x)
        augmention_data_y.append(y)
    if mix_ratio>0:
        # mixup
        idxs = np.random.choice(indices, int(x_len * mix_ratio), replace=False)
        augmentions_x, augmentions_y = mix_up(np.array(x)[idxs], np.array(y)[idxs])
        augmention_data_x.append(augmentions_x)
        augmention_data_y.append(augmentions_y)
    if ratio>0:
        #spec_augment
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = np.array([spec_augment(d) for d in np.array(x)[idxs]])
        augmention_data_x.append(augmentions_x)
        augmention_data_y.append(np.array(y)[idxs])

    x = np.concatenate(augmention_data_x, axis=0)
    y = np.concatenate(augmention_data_y, axis=0)
    x,y=shuffle(x,y)
    return x, y

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, labels, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                 win_length=400, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        'Initialization'
        self.dim = dim
        self.nfft = nfft
        self.spec_len = spec_len
        self.normalize = normalize
        self.mp_pooler = mp_pooler
        self.win_length = win_length
        self.hop_length = hop_length

        self.labels = labels
        self.shuffle = shuffle
        self.X = X
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_temp = [self.X[k] for k in indexes]
        X, y = self.__data_generation_mp(X_temp, indexes)
#         if self.augmentation:
#             X = X.squeeze()
#             X,y = get_augmention_data(self, X, y, ratio=0.5, mix_ratio=0, origin=True)
#             X = X[:,:,:,np.newaxis]
        
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_mp(self, X_temp, indexes):
        X = [self.mp_pooler.apply_async(ut.load_data, args=(mag, self.spec_len)) for mag in X_temp]
        X = np.expand_dims(np.array([p.get() for p in X]), -1)
        y = self.labels[indexes]
        return X, y

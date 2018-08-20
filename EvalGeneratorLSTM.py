import os
import keras
import keras_preprocessing
import keras_preprocessing.image as kimage
from keras_preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array
import keras.backend

import numpy as np
import librosa
import h5py

n_mels = 80
sr = 16000

def load_wave_to_mel(fname):
    hf_train1 = h5py.File(fname, 'r')
    mel = np.array(hf_train1.get('mel'))
    wave = np.array(hf_train1.get('wave'))
    sr = np.array(hf_train1.get('sampling_rate'))
    
    mel_len = 188
    
    if mel_len > mel.shape[1]:
        num_repeat = int(mel_len/mel.shape[1]) + 1
        melspect = np.tile(mel, num_repeat)
        melspect = melspect[:,:mel_len]
        melspect = np.rot90(melspect, )
        return melspect[np.newaxis, :,:mel_len]
    elif mel_len==mel.shape[1]:
        melspect = np.rot90(mel, )
        return melspect[np.newaxis, :,:mel_len]
    
    num_mel = int(mel.shape[1] / mel_len) + 1
    mel_tiled = np.tile(mel, 2)
    mel_tiled = mel_tiled[:, :num_mel*mel_len]
    
    mel_tiled = mel_tiled.reshape((mel_tiled.shape[0], num_mel, mel_len))
    
    res = np.zeros((num_mel,) + (mel_len, mel_tiled.shape[0]))
    for i in range(num_mel):
        res[i,:,:] =  np.rot90(mel_tiled[:,i,:], )
        
    return res

class JsonIterator(Iterator):
    def __init__(self, X, Y, image_data_generator, target_size=(256, 256)):
        
        batch_size = 1
        shuffle = False
        seed=None
        data_format='channels_last'
            
        self.X = X
        self.Y = Y
            
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.color_mode = 'grayscale'
        
        self.image_shape = self.target_size + (1,)
                
        self.classes_label = list(set(self.Y))
        self.classes_label.sort()
        
        self.class_mode = 'categorical'
        self.save_to_dir = None
        self.save_prefix = ''
        self.save_format = 'png'
        self.interpolation = 'nearest'

        self.subset = None

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        
        # First, count the number of samples and classes.
        self.samples = len(self.Y)
        self.num_classes = len(self.classes_label)
        
        self.class_indices = dict(zip(self.classes_label, range(self.num_classes)))

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        
        self.filenames = self.X
        self.classes = np.array([self.class_indices[y] for y in self.Y]).astype(np.int32)
        
        super(JsonIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=kimage.backend.floatx())
        
        fname = self.filenames[index_array[0]]
        x = load_wave_to_mel(fname)
        
        batch_y = np.zeros((1, self.num_classes), dtype=kimage.backend.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.
        
        return x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

class WaveDataGeneratorExtended(ImageDataGenerator):
    def flow_from_json(self, X, Y, target_size=(256, 256)):
        return JsonIterator(X, Y, self, target_size=target_size)
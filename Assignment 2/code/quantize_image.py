import numpy as np
from sklearn.cluster import KMeans


class ImageQuantizer:

    def __init__(self, b):
        self.b = b  # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour

        """
        H, W, _ = img.shape
        model = KMeans(n_clusters=2**self.b, n_init=3)
        reshaped_img = np.reshape(img, (H * W, 3))
        model.fit(reshaped_img)
        self.colours = model.cluster_centers_
        minimized_img = model.predict(reshaped_img)
        quantized_img = np.reshape(minimized_img, (H, W))
        return quantized_img

    def dequantize(self, quantized_img):
        H, W = quantized_img.shape
        img = np.zeros((H, W, 3), dtype='uint8')
        reshaped_img = quantized_img.flatten()
        flattened_img = self.colours[reshaped_img]
        # print(self.colours.shape)
        # print(reshaped_img.shape)
        print(flattened_img.shape)
        img = np.reshape(flattened_img, (H, W, 3))
        return img

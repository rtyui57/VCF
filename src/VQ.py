import gzip
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
import cv2
import PNG as EC
import main
import logging
import os

class CoDec(EC.CoDec):
    def encode(self):
        img = self.read()
        normalized_img = np.array(img,dtype=np.float64) / 255
        w, h, d = tuple(img.shape)
        img_reshaped = normalized_img.reshape((w*h, d))
        kmeans = KMeans(n_clusters=self.args.clusters, random_state=0)
        some_samples = shuffle(img_reshaped, random_state=0, n_samples=1_000)
        kmeans.fit(some_samples)
        centroids = kmeans.cluster_centers_
        labels = kmeans.predict(img_reshaped)
        with gzip.GzipFile("labels.npy", "wb") as f:
                    np.save(f, labels, True)
        with gzip.GzipFile("centroids.npy", "wb") as f:
                    np.save(f, centroids, True)
        with gzip.GzipFile("shape.npy", "wb") as f:
                    np.save(f, np.asarray([w,h]), True)
        cv2.imwrite(self.args.output, np.ones(shape=(4,4,3)))

    def decode(self):
        with gzip.GzipFile("labels.npy", "rb") as f:
                labels = np.load(f)
        with gzip.GzipFile("centroids.npy", "rb") as f:
                centroids = np.load(f)
        with gzip.GzipFile("shape.npy", "rb") as f:
                shape = np.load(f)
        os.remove("labels.npy")
        os.remove("centroids.npy")
        os.remove("shape.npy")
        k = labels.astype(np.uint8) # Up to 256 bins
        img = (centroids[k].reshape(shape[0], shape[1], -1) * 255).astype(np.uint8)
        self.write(img)

if __name__ == "__main__":
    EC.parser_encode.add_argument("-c", "--clusters", type=int, default=64)
    main.main(EC.parser, logging, CoDec)
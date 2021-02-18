import pickle
import unittest

import healpy as hp
import numpy as np
from pixell import enmap, utils


def generate_distance_healpix():
    np.random.seed(1234)
    nside = 128
    m = np.ones(12 * nside ** 2)
    idx = np.arange(12 * nside ** 2)
    for i in range(100):
        vec = hp.pix2vec(nside, np.random.choice(idx))
        disc = hp.query_disc(nside, vec, 5 / (60.0 * 180) * np.pi)
        m[disc] = 0.0
    dist = enmap.distance_transform_healpix(m)
    m[dist * 60 * 180 / np.pi < 5] = 0.0
    return np.asarray(m)


def generate_distance_car():
    np.random.seed(1234)
    shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj="car")
    m = enmap.ones(shape=shape, wcs=wcs)
    idx1 = np.random.randint(0, shape[0], 100)
    idx2 = np.random.randint(0, shape[1], 100)
    m[idx1, idx2] = 0.0
    dist = enmap.distance_transform(m)
    m[dist * 60 * 180 / np.pi < 5] = 0.0
    return np.asarray(m)


def store_data():
    d = {"car": generate_distance_car(), "healpix": generate_distance_healpix()}
    with open("./data/distances.pkl", "wb") as f:
        pickle.dump(d, f)


class DistanceTest(unittest.TestCase):
    def setUp(self):
        with open("data/distances.pkl", "rb") as f:
            self.ref = pickle.load(f)

    def test_distance_healpix(self):
        np.testing.assert_almost_equal(self.ref["healpix"], generate_distance_healpix(), decimal=7)

    def test_distance_car(self):
        np.testing.assert_almost_equal(self.ref["car"], generate_distance_car(), decimal=7)


if __name__ == "__main__":
    unittest.main()

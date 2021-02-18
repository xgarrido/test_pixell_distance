import pickle
import unittest

import numpy as np
from pixell import enmap, utils


def generate_distance_healpix():
    nside = 128
    m = np.ones(12 * nside ** 2)
    m[6 * nside ** 2] = 0.0
    dist = enmap.distance_transform_healpix(m)
    return np.asarray(dist)


def generate_distance_car():
    shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj="car")
    m = enmap.ones(shape=shape, wcs=wcs)
    m[250, 250] = 0.0
    dist = enmap.distance_transform(m)
    return np.asarray(dist)


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

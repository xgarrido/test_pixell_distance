import pickle
import unittest

import healpy as hp
import numpy as np
from pixell import enmap, utils

nholes = 10
hole_radius_arcmin = 20
apo_radius_degree = 1


def generate_distance_healpix():
    np.random.seed(14)
    nside = 256
    m = np.ones(12 * nside ** 2)
    idx = np.arange(12 * nside ** 2)
    for i in range(nholes):
        vec = hp.pix2vec(nside, np.random.choice(idx))
        disc = hp.query_disc(nside, vec, hole_radius_arcmin / (60.0 * 180) * np.pi)
        m[disc] = 0.0
    dist = enmap.distance_transform_healpix(m) * 180 / np.pi
    idx = np.where(dist > apo_radius_degree)
    m = 1 / 2 * (1 - np.cos(-np.pi * dist / apo_radius_degree))
    m[idx] = 1.0
    return np.asarray(m)


def generate_distance_car():
    np.random.seed(14)
    box = np.array([[-25, 25], [25, -25]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=5 * utils.arcmin, proj="car")
    m = enmap.ones(shape=shape, wcs=wcs)
    idx1 = np.random.randint(0, shape[0], nholes)
    idx2 = np.random.randint(0, shape[1], nholes)
    m[idx1, idx2] = 0.0
    dist = enmap.distance_transform(m) * 180 / np.pi
    idx = np.where(dist > apo_radius_degree)
    m = 1 / 2 * (1 - np.cos(-np.pi * dist / apo_radius_degree))
    m[idx] = 1.0
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

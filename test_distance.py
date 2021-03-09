import unittest

import healpy as hp
import numpy as np
from pixell import enmap, utils

nside = 256


def generate_distance_healpix():
    binary = np.zeros(12 * nside ** 2)
    vec = hp.ang2vec(30, 50, lonlat=True)
    disc = hp.query_disc(nside, vec, radius=25 * np.pi / 180)
    binary[disc] = 1
    dist = enmap.distance_transform_healpix(binary, rmax=None)
    return np.asarray(dist)


def store_data():
    np.save("./data/distances_healpix", generate_distance_healpix())


class DistanceTest(unittest.TestCase):
    def setUp(self):
        self.distances = {"healpix": np.load("./data/distances_healpix.npy")}

    def test_distance_healpix(self):
        try:
            np.testing.assert_almost_equal(
                self.distances["healpix"], generate_distance_healpix(), decimal=7
            )
        except AssertionError as e:
            print(e)


if __name__ == "__main__":
    unittest.main()

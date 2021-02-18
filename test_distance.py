import pickle
import unittest

import numpy as np
from pixell import enmap, utils

with open("data/distance_healpix.pkl", "rb") as f:
    ref_healpix = pickle.load(f)
with open("data/distance_car.pkl", "rb") as f:
    ref_car = pickle.load(f)


class DistanceTest(unittest.TestCase):
    def test_distance_healpix(self):
        nside = 128
        m = np.ones(12 * nside ** 2)
        m[6 * nside ** 2] = 0.0
        dist = enmap.distance_transform_healpix(m)

        np.testing.assert_almost_equal(ref_healpix, dist, decimal=7)

    def test_distance_healpix(self):
        shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj="car")
        m = enmap.ones(shape=shape, wcs=wcs)
        m[250, 250] = 0.0
        dist = enmap.distance_transform(m)

        np.testing.assert_almost_equal(ref_car, dist, decimal=7)


if __name__ == "__main__":
    unittest.main()

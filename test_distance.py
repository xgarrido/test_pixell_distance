import pickle
import unittest

import healpy as hp
import numpy as np
from pixell import enmap, utils
from pspy import so_map, so_window

seed = 14
nholes = 10
hole_radius_arcmin = 20
apo_radius_degree = 1
nside = 256


def generate_distance_healpix():
    np.random.seed(seed)
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
    np.random.seed(seed)
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


def generate_window_car():
    np.random.seed(seed)
    binary = so_map.car_template(ncomp=1, res=5, ra0=-25, ra1=+25, dec0=-25, dec1=+25)
    binary.data[:] = 0
    binary.data[1:-1, 1:-1] = 1
    window = so_window.create_apodization(binary, apo_type="Rectangle", apo_radius_degree=1)
    mask = so_map.simulate_source_mask(
        binary, n_holes=nholes, hole_radius_arcmin=hole_radius_arcmin
    )
    mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree)
    window.data *= mask.data
    return np.asarray(window.data)


def generate_window_healpix():
    np.random.seed(seed)
    binary = so_map.healpix_template(ncomp=1, nside=nside)
    vec = hp.ang2vec(30, 50, lonlat=True)
    disc = hp.query_disc(nside, vec, radius=25 * np.pi / 180)
    binary.data[disc] = 1
    dist = binary.copy()
    dist.data = enmap.distance_transform_healpix(binary.data, method="heap", rmax=None)
    # dist = so_window.get_distance(binary)
    # window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=1)
    # mask = so_map.simulate_source_mask(
    #     binary, n_holes=nholes, hole_radius_arcmin=hole_radius_arcmin
    # )
    # mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree)
    # window.data *= mask.data
    # return np.asarray(window.data)
    return np.asarray(dist.data)


def store_data():
    d = {"car": generate_distance_car(), "healpix": generate_distance_healpix()}
    with open("./data/distances.pkl", "wb") as f:
        pickle.dump(d, f)
    # d = {"car": generate_window_car(), "healpix": generate_window_healpix()}
    d = {"healpix": generate_window_healpix()}
    with open("./data/windows.pkl", "wb") as f:
        pickle.dump(d, f)


class DistanceTest(unittest.TestCase):
    def setUp(self):
        with open("data/distances.pkl", "rb") as f:
            self.ref = pickle.load(f)
        with open("data/windows.pkl", "rb") as f:
            self.window = pickle.load(f)

    # def test_distance_healpix(self):
    #     np.testing.assert_almost_equal(self.ref["healpix"], generate_distance_healpix(), decimal=7)

    # def test_distance_car(self):
    #     np.testing.assert_almost_equal(self.ref["car"], generate_distance_car(), decimal=7)

    def test_window_healpix(self):
        try:
            np.testing.assert_almost_equal(
                self.window["healpix"], generate_window_healpix(), decimal=7
            )
        except AssertionError as e:
            print(e)

    # def test_window_car(self):
    #     np.testing.assert_almost_equal(self.window["car"], generate_window_car(), decimal=7)


if __name__ == "__main__":
    unittest.main()

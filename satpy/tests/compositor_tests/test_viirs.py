#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for VIIRS compositors."""

import unittest


class TestVIIRSComposites(unittest.TestCase):
    """Test VIIRS-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites.config_loader import CompositorLoader
        cl = CompositorLoader()
        cl.load_sensor_composites('viirs')

    def test_histogram_dnb(self):
        """Test the 'histogram_dnb' compositor."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.viirs import HistogramDNB
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = HistogramDNB('histogram_dnb', prerequisites=('dnb',),
                            standard_name='toa_outgoing_radiance_per_'
                                          'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        # data changes by row, sza changes by col for testing
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        res = comp((c01, c02))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'histogram_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        unique_values = np.unique(data)
        np.testing.assert_allclose(unique_values, [0.5994, 0.7992, 0.999], rtol=1e-3)

    def test_adaptive_dnb(self):
        """Test the 'adaptive_dnb' compositor."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.viirs import AdaptiveDNB
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = AdaptiveDNB('adaptive_dnb', prerequisites=('dnb',),
                           standard_name='toa_outgoing_radiance_per_'
                                         'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        res = comp((c01, c02))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'adaptive_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        np.testing.assert_allclose(data.data, 0.999, rtol=1e-4)

    def test_erf_dnb(self):
        """Test the 'dynamic_dnb' or ERF DNB compositor."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.viirs import ERFDNB
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = ERFDNB('dynamic_dnb', prerequisites=('dnb',),
                      standard_name='toa_outgoing_radiance_per_'
                                    'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        lza = np.zeros((rows, cols)) + 70.0
        lza[:, 3] += 20.0
        lza[:, 4:] += 45.0
        lza = da.from_array(lza, chunks=25)
        c03 = xr.DataArray(lza,
                           dims=('y', 'x'),
                           attrs={'name': 'lunar_zenith_angle', 'area': area})
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'dynamic_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [0.00000000e+00, 1.00446703e-01, 1.64116082e-01, 2.09233451e-01,
                                            1.43916324e+02, 2.03528498e+02, 2.49270516e+02])

    def test_hncc_dnb(self):
        """Test the 'hncc_dnb' compositor."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.viirs import NCCZinke
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = NCCZinke('hncc_dnb', prerequisites=('dnb',),
                        standard_name='toa_outgoing_radiance_per_'
                                      'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        lza = np.zeros((rows, cols)) + 70.0
        lza[:, 3] += 20.0
        lza[:, 4:] += 45.0
        lza = da.from_array(lza, chunks=25)
        c03 = xr.DataArray(lza,
                           dims=('y', 'x'),
                           attrs={'name': 'lunar_zenith_angle', 'area': area})
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'hncc_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'ncc_radiance')
        data = res.compute()
        unique = np.unique(data)
        np.testing.assert_allclose(
            unique, [3.48479712e-04, 6.96955799e-04, 1.04543189e-03, 4.75394738e-03,
                     9.50784532e-03, 1.42617433e-02, 1.50001560e+03, 3.00001560e+03,
                     4.50001560e+03])

# Copyright (c) 2010-2023 Satpy developers
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
"""Unit tests for Scene conversion functionality."""
from __future__ import annotations

import contextlib
import sys
from datetime import datetime
from importlib import import_module, reload
from unittest import mock

import pytest
import xarray as xr
from dask import array as da

from satpy import Scene

try:
    from datatree import DataTree
except ImportError:
    DataTree = None


# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


@pytest.mark.usefixtures("include_test_etc")
class TestSceneSerialization:
    """Test the Scene serialization."""

    def test_serialization_with_readers_and_data_arr(self):
        """Test that dask can serialize a Scene with readers."""
        from distributed.protocol import deserialize, serialize

        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        cloned_scene = deserialize(*serialize(scene))
        assert scene._readers.keys() == cloned_scene._readers.keys()
        assert scene.all_dataset_ids == scene.all_dataset_ids


class TestSceneConversions:
    """Test Scene conversion to geoviews, xarray, etc."""

    def test_to_xarray_dataset_with_empty_scene(self):
        """Test converting empty Scene to xarray dataset."""
        scn = Scene()
        xrds = scn.to_xarray_dataset()
        assert isinstance(xrds, xr.Dataset)
        assert len(xrds.variables) == 0
        assert len(xrds.coords) == 0

    def test_geoviews_basic_with_area(self):
        """Test converting a Scene to geoviews with an AreaDefinition."""
        from pyresample.geometry import AreaDefinition
        scn = Scene()
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              2, 2, [-200, -200, 200, 200])
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None

    def test_geoviews_basic_with_swath(self):
        """Test converting a Scene to geoviews with a SwathDefinition."""
        from pyresample.geometry import SwathDefinition
        scn = Scene()
        lons = xr.DataArray(da.zeros((2, 2)))
        lats = xr.DataArray(da.zeros((2, 2)))
        area = SwathDefinition(lons, lats)
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None


def test_to_datatree_no_datatree(monkeypatch):
    """Test that datatree not being installed causes an exception with a nice message."""
    with make_module_unimportable("datatree", "satpy.scene"):
        scn = Scene()
        with pytest.raises(ImportError, match='xarray-datatree'):
            scn.to_xarray_datatree()


@contextlib.contextmanager
def make_module_unimportable(module_name_to_remove: str, module_doing_the_importing: str):
    """Mock sys.modules to make a module unimportable from another module."""
    with mock.patch.dict(sys.modules):
        sys.modules[module_name_to_remove] = None  # type: ignore[assignment]
        if module_doing_the_importing in sys.modules:
            reload(sys.modules[module_doing_the_importing])
        else:
            import_module(module_doing_the_importing)
        yield None


@pytest.mark.skipif(DataTree is None, reason="Optional 'xarray-datatree' library is not installed")
class TestToDataTree:
    """Test Scene conversion to an xarray DataTree."""

    def test_empty_scene(self):
        """Test that an empty Scene can be converted to a DataTree."""
        from datatree import DataTree

        scn = Scene()
        data_tree = scn.to_xarray_datatree()
        assert isinstance(data_tree, DataTree)
        assert len(data_tree) == 0

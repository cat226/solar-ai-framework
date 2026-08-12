from datetime import datetime, timezone

import pytest

from services.physics import calculate_cloud_factor, compute_physics

NOON = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


def test_cloud_factor_boundaries():
    assert calculate_cloud_factor(0.0) == 1.0
    assert calculate_cloud_factor(50.0) == pytest.approx(0.5)
    assert calculate_cloud_factor(90.0) == pytest.approx(0.1)
    assert calculate_cloud_factor(100.0) == 0.0


def test_cloud_factor_clamps_out_of_range_inputs():
    assert calculate_cloud_factor(-10.0) == 1.0
    assert calculate_cloud_factor(110.0) == 0.0


def test_heavy_clouds_reduce_noon_irradiance_below_300():
    result = compute_physics(18.0, 2.0, 90.0, "Clean", 0.0, 0.0, NOON)
    assert result.cloud_factor == pytest.approx(0.1)
    assert result.irradiance_wm2 == pytest.approx(100.0)
    assert result.irradiance_wm2 < 300.0


def test_full_cloud_cover_blocks_direct_irradiance():
    result = compute_physics(18.0, 2.0, 100.0, "Clean", 0.0, 0.0, NOON)
    assert result.cloud_factor == 0.0
    assert result.irradiance_wm2 == 0.0

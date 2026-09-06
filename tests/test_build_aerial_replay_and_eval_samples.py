"""Tests for training/detection/build_aerial_replay_sample.py and
build_aerial_eval_sample.py - the two scripts that draw deterministic,
disjoint samples from the production aerial dataset for the personal
domain-adaptation experiment (replay/monitor from TRAIN only, eval
sample from TEST only, so training/monitoring and final evaluation can
never share an image)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from training.detection.build_aerial_replay_sample import build_replay_sample
from training.detection.build_aerial_eval_sample import build_eval_sample


def _make_split(root: Path, n: int) -> Path:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(images_dir / f"img{i:04d}.png")
        (labels_dir / f"img{i:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


class TestBuildReplaySample:
    def test_replay_and_monitor_are_disjoint(self, tmp_path):
        source = _make_split(tmp_path / "aerial_train", 50)
        out = tmp_path / "out"
        manifest = build_replay_sample(source, out, replay_count=20, monitor_count=10, seed=1)
        assert set(manifest["replay_stems"]).isdisjoint(manifest["monitor_stems"])
        assert len(manifest["replay_stems"]) == 20
        assert len(manifest["monitor_stems"]) == 10

    def test_output_files_copied_correctly(self, tmp_path):
        source = _make_split(tmp_path / "aerial_train", 30)
        out = tmp_path / "out"
        manifest = build_replay_sample(source, out, replay_count=10, monitor_count=5, seed=2)
        for stem in manifest["replay_stems"]:
            assert (out / "replay" / "images" / f"{stem}.png").is_file()
            assert (out / "replay" / "labels" / f"{stem}.txt").is_file()
        for stem in manifest["monitor_stems"]:
            assert (out / "monitor" / "images" / f"{stem}.png").is_file()

    def test_deterministic_across_runs(self, tmp_path):
        source = _make_split(tmp_path / "aerial_train", 40)
        m1 = build_replay_sample(source, tmp_path / "out1", replay_count=15, monitor_count=5, seed=99)
        m2 = build_replay_sample(source, tmp_path / "out2", replay_count=15, monitor_count=5, seed=99)
        assert m1["replay_stems"] == m2["replay_stems"]
        assert m1["monitor_stems"] == m2["monitor_stems"]

    def test_different_seeds_give_different_samples(self, tmp_path):
        source = _make_split(tmp_path / "aerial_train", 40)
        m1 = build_replay_sample(source, tmp_path / "out1", replay_count=15, monitor_count=5, seed=1)
        m2 = build_replay_sample(source, tmp_path / "out2", replay_count=15, monitor_count=5, seed=2)
        assert m1["replay_stems"] != m2["replay_stems"]

    def test_insufficient_images_raises(self, tmp_path):
        source = _make_split(tmp_path / "aerial_train", 10)
        with pytest.raises(ValueError):
            build_replay_sample(source, tmp_path / "out", replay_count=20, monitor_count=5, seed=1)


class TestBuildEvalSample:
    def test_eval_sample_never_touches_train(self, tmp_path):
        """The eval sample function only ever reads from the path it's
        given - passing a TEST-split directory never reads any TRAIN
        data, by construction (no shared code path with build_replay_sample)."""
        test_source = _make_split(tmp_path / "aerial_test", 30)
        out = tmp_path / "eval_out"
        manifest = build_eval_sample(test_source, out, count=10, seed=5)
        assert manifest["count"] == 10
        assert manifest["full_test_split_size"] == 30
        for stem in manifest["stems"]:
            assert (out / "images" / f"{stem}.png").is_file()

    def test_deterministic(self, tmp_path):
        source = _make_split(tmp_path / "aerial_test", 25)
        m1 = build_eval_sample(source, tmp_path / "out1", count=8, seed=7)
        m2 = build_eval_sample(source, tmp_path / "out2", count=8, seed=7)
        assert m1["stems"] == m2["stems"]

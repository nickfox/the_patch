"""Tests for argument parsing in mlxmas.test_cross."""

import pytest

from mlxmas.test_cross import create_parser


class TestCreateParser:

    def test_defaults(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.procrustes is None
        assert args.start_layer == 12
        assert args.sender_layer == 13
        assert args.latent_steps == 10
        assert args.max_tokens == 2048
        assert args.temperature == 0.6

    def test_procrustes_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--procrustes", "alignment.npz"])
        assert args.procrustes == "alignment.npz"

    def test_start_layer(self):
        parser = create_parser()
        args = parser.parse_args(["--start-layer", "4"])
        assert args.start_layer == 4

    def test_sender_layer(self):
        parser = create_parser()
        args = parser.parse_args(["--sender-layer", "24"])
        assert args.sender_layer == 24

    def test_latent_steps(self):
        parser = create_parser()
        args = parser.parse_args(["--latent-steps", "5"])
        assert args.latent_steps == 5

    def test_combined_args(self):
        parser = create_parser()
        args = parser.parse_args([
            "--procrustes", "foo.npz",
            "--start-layer", "6",
            "--sender-layer", "30",
            "--temperature", "0.8",
        ])
        assert args.procrustes == "foo.npz"
        assert args.start_layer == 6
        assert args.sender_layer == 30
        assert args.temperature == 0.8

"""
Adversarial / edge case tests.

The contract here is *no crashes*.  Detection quality is allowed to degrade
on these inputs - the model just has to keep its mouth shut and return a
valid Results object.
"""

import os

import cv2
import numpy as np
import pytest

import config


def _model():
    """Try to load the active model weights; skip the whole module otherwise."""
    weights = config.DEFAULT_WEIGHTS
    if not os.path.exists(weights):
        pytest.skip(f"Weights not available: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed")
    try:
        return YOLO(weights)
    except Exception as e:
        pytest.skip(f"Could not load weights: {e}")


@pytest.fixture(scope="module")
def model():
    return _model()


def _run(model, img):
    """Run a prediction and verify it returned a list-like Results."""
    out = model.predict(
        source=img, conf=0.25, iou=0.5, imgsz=640,
        max_det=50, verbose=False,
    )
    assert out is not None
    assert len(out) >= 1


class AdversarialTester:
    @staticmethod
    def create_test_image(width: int, height: int, fill=(128, 128, 128)) -> np.ndarray:
        """Solid-color BGR canvas."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = fill[::-1]  # RGB -> BGR
        return img


class TestAdversarial:
    def test_extreme_brightness(self, model):
        img = AdversarialTester.create_test_image(640, 480, (255, 255, 255))
        _run(model, img)

    def test_extreme_darkness(self, model):
        img = AdversarialTester.create_test_image(640, 480, (10, 10, 10))
        _run(model, img)

    def test_high_noise(self, model):
        img = AdversarialTester.create_test_image(640, 480, (128, 128, 128))
        noise = np.random.randint(0, 80, img.shape, dtype=np.uint8)
        noisy = cv2.add(img, noise)
        _run(model, noisy)

    def test_motion_blur(self, model):
        img = AdversarialTester.create_test_image(640, 480)
        ksize = 15
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        blurred = cv2.filter2D(img, -1, kernel)
        _run(model, blurred)

    def test_rotated_90_degrees(self, model):
        img = AdversarialTester.create_test_image(640, 480)
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        _run(model, rotated)

    def test_scaled_down_in_canvas(self, model):
        canvas = AdversarialTester.create_test_image(416, 416, (255, 255, 255))
        tile = AdversarialTester.create_test_image(100, 100, (50, 100, 150))
        canvas[10:110, 10:110] = tile
        _run(model, canvas)

    def test_unusual_aspect_ratio(self, model):
        img = AdversarialTester.create_test_image(1280, 200)
        _run(model, img)

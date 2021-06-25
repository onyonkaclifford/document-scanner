import copy
import os

import cv2
import numpy as np
import pytest

from document_scanner import DocumentScanner


@pytest.fixture
def image_path():
    return os.path.join("test_data", "img.png")


@pytest.fixture
def images():
    img_bgr = cv2.imread(os.path.join("test_data", "img.png"))
    img_rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb


@pytest.fixture
def edged_images(images):
    def get_edged_image(img):
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged_image = cv2.Canny(blurred, 75, 200)
        return edged_image

    img_bgr, img_rgb = images
    return get_edged_image(img_bgr), get_edged_image(img_rgb)


@pytest.fixture
def contours(edged_images):
    def get_contours(edged_img):
        contours = cv2.findContours(
            edged_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 2:
            resultant_contours = contours[0]
        elif len(contours) == 3:  # Support for some versions of cv2
            resultant_contours = contours[1]
        else:
            raise Exception

        return resultant_contours

    edged_img_bgr, edged_img_rgb = edged_images
    return get_contours(edged_img_bgr), get_contours(edged_img_rgb)


@pytest.fixture
def document_outlines(contours):
    def get_outline(cnts):
        """
        :param cnts: contours
        """
        cnts_sorted = sorted(copy.deepcopy(cnts), key=cv2.contourArea, reverse=True)
        document_outline = None

        for cnt in cnts_sorted:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                document_outline = approx
                break

        return document_outline

    contours_bgr, contours_rgb = contours
    return get_outline(contours_bgr), get_outline(contours_rgb)


def test_get_image(image_path):
    img_bgr, img_rgb = DocumentScanner.get_image(image_path)
    assert isinstance(img_bgr, np.ndarray)
    assert isinstance(img_rgb, np.ndarray)


def test_get_image__resized(image_path):
    img_bgr, img_rgb = DocumentScanner.get_image(image_path, 60)

    assert isinstance(img_bgr, np.ndarray)
    assert isinstance(img_rgb, np.ndarray)

    assert img_bgr.shape[1] == 60
    assert img_rgb.shape[1] == 60


def test_get_image__empty_image_path():
    with pytest.raises(AttributeError):
        DocumentScanner.get_image("")

    with pytest.raises(AttributeError):
        DocumentScanner.get_image("", 60)


def test_get_processed_image(images):
    img_bgr, img_rgb = images

    img_processed_bgr = DocumentScanner.get_processed_image(img_bgr)
    assert isinstance(img_processed_bgr, np.ndarray)

    img_processed_rgb = DocumentScanner.get_processed_image(img_rgb)
    assert isinstance(img_processed_rgb, np.ndarray)


def test_get_text(images):
    img_bgr, img_rgb = images

    text_bgr = DocumentScanner.get_text(img_bgr)
    assert isinstance(text_bgr, str)

    text_rgb = DocumentScanner.get_text(img_rgb)
    assert isinstance(text_rgb, str)


def test_get_edged_image(images):
    img_bgr, img_rgb = images

    edged_img_bgr = DocumentScanner.get_edged_image(img_bgr)
    assert isinstance(edged_img_bgr, np.ndarray)

    edged_img_rgb = DocumentScanner.get_edged_image(img_rgb)
    assert isinstance(edged_img_rgb, np.ndarray)


def test_get_contours(edged_images):
    edged_img_bgr, edged_img_rgb = edged_images

    contours_bgr = DocumentScanner.get_contours(edged_img_bgr)
    assert isinstance(contours_bgr, list)

    contours_rgb = DocumentScanner.get_contours(edged_img_rgb)
    assert isinstance(contours_rgb, list)


def test_get_document_outline(contours):
    contours_bgr, contours_rgb = contours

    document_outline_bgr = DocumentScanner.get_document_outline(contours_bgr)
    assert isinstance(document_outline_bgr, np.ndarray)

    document_outline_rgb = DocumentScanner.get_document_outline(contours_rgb)
    assert isinstance(document_outline_rgb, np.ndarray)


def test_get_aligned_document(images, document_outlines):
    img_bgr, img_rgb = images
    document_outline_bgr, document_outline_rgb = document_outlines

    aligned_bgr = DocumentScanner.get_aligned_document(img_bgr, document_outline_bgr)
    assert isinstance(aligned_bgr, np.ndarray)

    aligned_rgb = DocumentScanner.get_aligned_document(img_rgb, document_outline_rgb)
    assert isinstance(aligned_rgb, np.ndarray)

import cv2
import numpy as np
import pytesseract
from imutils.perspective import four_point_transform


class DocumentScanner:
    @staticmethod
    def get_image(image_path: str, resize_width: int = None):
        """
        Get image array from image path

        :param image_path: location of image
        :param resize_width: if not None image is resized to this width, else no resize
        :return: tuple of 2 image arrays, (bgr image, rgb image)
        """
        image = cv2.imread(image_path)
        if resize_width is not None:
            h, w = image.shape[:2]
            ratio = resize_width / float(w)
            if resize_width < w:
                image = cv2.resize(
                    image, (resize_width, int(h * ratio)), interpolation=cv2.INTER_AREA
                )
            elif resize_width > w:
                image = cv2.resize(
                    image,
                    (resize_width, int(h * ratio)),
                    interpolation=cv2.INTER_LINEAR,
                )
        return image, cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    @staticmethod
    def get_processed_image(image: np.ndarray):
        """
        Combines all processing steps to ready image for text extraction.
        Returned image retains colour channels format of image passed as argument.

        :param image: image to be processed
        :return: processed image ready for text extraction
        """
        edged_image = DocumentScanner.get_edged_image(image, (5, 5), 75, 200)
        contours = DocumentScanner.get_contours(edged_image)
        document_outline = DocumentScanner.get_document_outline(contours)
        return DocumentScanner.get_aligned_document(image, document_outline)

    @staticmethod
    def get_text(aligned_image_rgb: np.ndarray, config="--psm 4"):
        """
        Text extraction. Colour channels format of image passed needs to be rgb.

        :param aligned_image_rgb: processed image that's ready for text extraction
        :param config: pytesseract image_to_string config
        :return: extracted text
        """
        return pytesseract.image_to_string(
            cv2.cvtColor(aligned_image_rgb, cv2.COLOR_BGR2RGB), config=config
        )

    @staticmethod
    def get_edged_image(image: np.ndarray, ksize=(5, 5), threshold1=75, threshold2=200):
        """
        Highlight edges in the image

        :param image: image to be edged
        :param ksize: gaussian blur kernel size
        :param threshold1: Canny algorithm value for edge linking
        :param threshold2: Canny algorithm value for finding initial segments of strong edges
        :return: edged image
        """
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, ksize, 0)
        return cv2.Canny(blurred, threshold1, threshold2)

    @staticmethod
    def get_contours(edged_image: np.ndarray):
        """
        Returns contours

        :param edged_image: image that's been edged
        :return: contours
        """
        contours = cv2.findContours(
            edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 2:
            return contours[0]
        elif len(contours) == 3:  # Support for some versions of cv2
            return contours[1]

    @staticmethod
    def get_document_outline(contours):
        """
        Returns the outline of the document's edge

        :param contours: contours within the image
        :return: document outline
        """
        contours = sorted(contours.copy(), key=cv2.contourArea, reverse=True)
        document_outline = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                document_outline = approx
                break
        return document_outline

    @staticmethod
    def get_aligned_document(image: np.ndarray, document_outline):
        """
        Crops the document from the background and aligns it orthogonally to the viewer

        :param image: original image
        :param document_outline: outline of the document
        :return: aligned document
        """
        return four_point_transform(image.copy(), document_outline.copy().reshape(4, 2))

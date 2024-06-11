import cv2
import numpy as np


class ObjectDetector:
    """
    A class to detect objects in an image using a Haar Cascade classifier and detect the color of the largest object.

    Attributes:
        detector (cv2.CascadeClassifier): The Haar Cascade classifier for object detection.
        scaleFactor (float): Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors (int): Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        minSize (tuple): Minimum possible object size. Objects smaller than this are ignored.
        label_dict (dict): Dictionary to resolve integers to corresponding color labels.

    Author:
        - Luca Stanger

    """

    def __init__(self, haarcascade_path, label_dict, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        Initializes the ObjectDetector with a Haar Cascade file and detection parameters.

        Args:
            haarcascade_path (str): Path to the Haar Cascade file.
            label_dict (dict): Dictionary to resolve integers to corresponding color labels.
            scaleFactor (float, optional): Scale factor for the image pyramid. Defaults to 1.1.
            minNeighbors (int, optional): Minimum number of neighbors for a rectangle to be retained. Defaults to 5.
            minSize (tuple, optional): Minimum size of the detected objects. Defaults to (30, 30).

        Raises:
            ValueError: If the Haar Cascade file cannot be loaded.
        """
        # Load the Haar Cascade file
        self.detector = cv2.CascadeClassifier(haarcascade_path)
        if self.detector.empty():
            raise ValueError(
                f"Failed to load Haar Cascade from {haarcascade_path}")

        # Set detection parameters
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

        # Set label dictionary
        self.label_dict = label_dict

    def get_largest_object(self, image_path):
        """
        Detects objects in an image and returns the largest detected object.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Coordinates of the largest object in the format (x, y, w, h) and the original image.

        Raises:
            ValueError: If the image cannot be loaded.

        Example:
            >>> detector = ObjectDetector('haarcascade_frontalface_default.xml', label_dict)
            >>> largest_object, image = detector.get_largest_object('path_to_image.jpg')
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect objects
        objects = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )

        if len(objects) == 0:
            return None, image

        # Find the largest object
        largest_object = max(objects, key=lambda rect: rect[2] * rect[3])

        return largest_object, image

    def get_cropped_image(self, image, largest_object):
        """
        Crops the image to the largest detected object.

        Args:
            image (ndarray): The original image.
            largest_object (tuple): Coordinates of the largest object in the format (x, y, w, h).

        Returns:
            ndarray: Cropped image of the largest object. None if no object is provided.

        Example:
            >>> detector = ObjectDetector('haarcascade_frontalface_default.xml', label_dict)
            >>> largest_object, image = detector.get_largest_object('path_to_image.jpg')
            >>> cropped_image = detector.get_cropped_image(image, largest_object)
        """
        if largest_object is None:
            return None

        x, y, w, h = largest_object
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image

    def detect_color(self, image):
        """
        Detects the color of the largest object in the image.

        Args:
            image (ndarray): The image containing the detected object.

        Returns:
            str: Label of the detected color.

        Example:
            >>> detector = ObjectDetector('haarcascade_frontalface_default.xml', label_dict)
            >>> largest_object, image = detector.get_largest_object('path_to_image.jpg')
            >>> color = detector.detect_color(image)
        """
        image = cv2.resize(image, (100, 100))

        # Assuming image is always 100x100 pixels
        color = np.mean(image[20:50, 60:80], axis=(0, 1))
        color = cv2.cvtColor(
            np.uint8([[[color[0], color[1], color[2]]]]), cv2.COLOR_BGR2HLS)[0][0]
        hue = color[0]  # range 0-180
        sat = color[2]  # range 0-255

        print(color, hue, sat)

        return self.get_color(hue, sat)

    def get_color(self, hue: int, sat: int) -> str:
        """
        Determines the color based on hue and saturation values.

        Args:
            hue (int): The hue value (0-180).
            sat (int): The saturation value (0-255).

        Returns:
            str: Label of the detected color.

        Example:
            >>> detector = ObjectDetector('haarcascade_frontalface_default.xml', label_dict)
            >>> color_label = detector.get_color(90, 100)
        """
        if 85 <= hue < 130 and sat >= 40:
            return self.label_dict[3]  # Blue
        elif (130 <= hue <= 180 or 0 <= hue < 15) and sat >= 40:
            return self.label_dict[2]  # Red
        else:
            return self.label_dict[1]  # White

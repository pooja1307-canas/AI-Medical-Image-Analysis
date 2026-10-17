from PIL import Image
import numpy as np
from skimage.feature import hog

IMAGE_SIZE = (128, 128)

def image_to_features(image):
    """Convert a PIL image to a compact HOG feature vector for a baseline model."""
    image = image.convert("L").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    features = hog(
        array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.reshape(1, -1)

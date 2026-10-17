from PIL import Image
from src.image_utils import image_to_features

def test_feature_extractor_returns_single_feature_row():
    image = Image.new("RGB", (64, 64), color="white")
    features = image_to_features(image)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert features.shape[1] > 0

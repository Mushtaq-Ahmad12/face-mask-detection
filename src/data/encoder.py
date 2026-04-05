from sklearn.preprocessing import LabelEncoder
import numpy as np

class MaskLabelEncoder:
    """
    A utility wrapper around scikit-learn's LabelEncoder 
    specifically tailored for Face Mask classification labels.
    """
    def __init__(self):
        self.encoder = LabelEncoder()
        
    def fit(self, labels: list[str]) -> None:
        """Fit the encoder to the given labels."""
        self.encoder.fit(labels)
        
    def transform(self, labels: list[str]) -> np.ndarray:
        """Transform labels to normalized encodings."""
        return self.encoder.transform(labels)
        
    def fit_transform(self, labels: list[str]) -> np.ndarray:
        """Fit the encoder and return transformed labels."""
        return self.encoder.fit_transform(labels)
        
    def inverse_transform(self, encoded_labels: list[int]) -> np.ndarray:
        """Transform encoded labels back to original strings."""
        return self.encoder.inverse_transform(encoded_labels)
        
    def get_classes(self) -> np.ndarray:
        """Return the mapped classes after fitting."""
        return self.encoder.classes_

# Helper generic function version
def encode_labels(labels: list[str]) -> tuple[np.ndarray, LabelEncoder]:
    """
    Stand-alone helper to efficiently encode a list of strings into integers.
    Returns the encoded array and the fitted LabelEncoder object.
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

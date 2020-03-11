from .cnn_wrapper import CnnWrapper
from.classifier import Classifier


class ClassifierFactory:
    """
    Factory class for creation of multiple classification models
    """

    @staticmethod
    def get_classifier_wrapper(*args, **kwargs) -> Classifier:
        """
        Method thath creates classifier
        """
        return CnnWrapper()

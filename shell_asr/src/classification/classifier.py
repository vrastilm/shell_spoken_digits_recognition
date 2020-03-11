from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    Abstract classifier method
    """

    @abstractmethod
    def classify(self, X):
        """
        Method for classification
        """

        pass

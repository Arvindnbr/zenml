
import os
import tempfile
from typing import Any, ClassVar, Type
from scripts.utils.common import get_highest_train_folder
from ultralytics import YOLO
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
    PyTorchModuleMaterializer,
)
from zenml.io import fileio

DEFAULT_FILENAME = "objdet.pt"


class UltralyticsMaterializer(PyTorchModuleMaterializer):
    """Base class for ultralytics YOLO models."""

    FILENAME: ClassVar[str] = DEFAULT_FILENAME
    SKIP_REGISTRATION: ClassVar[bool] = True
    ASSOCIATED_TYPES = (YOLO,)

    def load(self, data_type: Type[Any]) -> YOLO:
        """Reads a ultralytics YOLO model from a serialized JSON file.

        Args:
            data_type: A ultralytics YOLO type.

        Returns:
            A ultralytics YOLO object.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)

        # Create a temporary folder
        with tempfile.TemporaryDirectory(prefix="zenml-temp-") as temp_dir:
            temp_file = os.path.join(str(temp_dir), DEFAULT_FILENAME)

            # Copy from artifact store to temporary file
            fileio.copy(filepath, temp_file)
            model = YOLO(temp_file)

            return model

    def save(self, model: YOLO) -> None:
        """Creates a JSON serialization for a ultralytics YOLO model.

        Args:
            model: A ultralytics YOLO model.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        train = get_highest_train_folder("runs/detect")
        modelpath = f"runs/detect/{train}/weights/best.pt"

        fileio.copy(modelpath, filepath)

import os
import tempfile
from typing import Any, ClassVar, Type
from scripts.config.configuration import ConfigurationManager
from ultralytics import YOLO
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
    PyTorchModuleMaterializer,
)
from zenml.io import fileio

DEFAULT_FILENAME = "YOLOdet.pt"

config = ConfigurationManager()
trainconfig = config.get_train_log_config()

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
        project_root = f"{trainconfig.runs_root}/{trainconfig.experiment_name}/{trainconfig.model_name}"
        modelpath = f"{project_root}/weights/best.pt"

        fileio.copy(modelpath, filepath)
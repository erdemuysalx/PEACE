"""Object detection backends — Strategy pattern for YOLO and VLM detectors."""

import base64
import json
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from ollama import Client
from pydantic import ValidationError

from peace.exceptions import InferenceError
from peace.schemas import Object


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface for all object detection backends."""

    def detect(self, image: np.ndarray) -> list[Object]:
        """Run detection on a RGB image and return detected objects."""
        ...

    @property
    def is_ready(self) -> bool:
        """Return True if the detector is initialised and ready to use."""
        ...

    @property
    def error_message(self) -> str | None:
        """Return the initialisation error message, or None if the detector is ready."""
        ...


class YoloDetector:
    """YOLO detector backed by the Ultralytics library.

    Model weights are auto-downloaded on first use when given a model name
    such as 'yolov8n.pt'. Alternatively pass a local path to a .pt file.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Load the YOLO model weights; sets _error if loading fails."""
        self._conf = confidence_threshold
        self._device = device
        self._model: Any = None
        self._error: str | None = None

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
        except Exception as e:
            self._error = str(e)

    @property
    def is_ready(self) -> bool:
        """Return True if the YOLO model loaded successfully."""
        return self._model is not None

    @property
    def error_message(self) -> str | None:
        """Return the model load error, or None if the model loaded successfully."""
        return self._error

    def detect(self, image: np.ndarray) -> list[Object]:
        """Run YOLO inference on a RGB image and return detected objects."""
        if not self.is_ready:
            return []

        results = self._model.predict(
            image, conf=self._conf, device=self._device, verbose=False
        )[0]

        objects: list[Object] = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append(
                Object(
                    id=i,
                    label=results.names[int(box.cls[0])],
                    bbox=[x1, y1, x2, y2],
                    confidence=round(float(box.conf[0]) * 100, 1),
                )
            )

        return objects


class VlmDetector:
    """Vision-Language Model detector using Ollama structured output."""

    def __init__(
        self,
        client: Client,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        num_predict: int = 800,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Store Ollama client, model config, and probe availability."""
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._num_predict = num_predict
        self._confidence_threshold = confidence_threshold
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Return True if the configured vision model is available on the Ollama server."""
        try:
            names = [m.model for m in self._client.list().models]
            return any(self._model in name for name in names)
        except Exception:
            return False

    @property
    def is_ready(self) -> bool:
        """Return True if the VLM was found on the Ollama server at startup."""
        return self._available

    @property
    def error_message(self) -> str | None:
        """Return None — VlmDetector does not capture a model load error."""
        return None

    def detect(self, image: np.ndarray) -> list[Object]:
        """Send a RGB image to the VLM and return validated, filtered detections."""
        if not self.is_ready:
            return []

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": self._system_prompt,
                    "images": [image_b64],
                }
            ],
            # think= True,
            format={
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": Object.model_json_schema(),
                    }
                },
                "required": ["objects"],
            },
            options={
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        )
        try:
            raw = json.loads(response.message.content)
            objects = [Object(**o) for o in raw.get("objects", [])]
        except (json.JSONDecodeError, ValidationError) as e:
            raise InferenceError(f"VLM returned unparseable detection output: {e}") from e
        return self._filter(objects)

    def _filter(self, objects: list[Object]) -> list[Object]:
        """Remove detections with invalid or implausibly large bounding boxes."""
        valid: list[Object] = []
        for obj in objects:
            if len(obj.bbox) != 4:
                continue
            x1, y1, x2, y2 = obj.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            # Reject bboxes that cover >95% of a full-HD frame (VLM hallucination)
            if (x2 - x1) * (y2 - y1) > 0.95 * 1920 * 1080:
                continue
            valid.append(obj)
        return valid

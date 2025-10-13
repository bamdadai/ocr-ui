import pickle
from typing import List, Any
import cv2
import numpy as np
from redis import Redis
from app.core.config import settings

# A Redis client instance can be shared.
# It manages its own connection pool internally.
redis_client = Redis.from_url(settings.CELERY_BACKEND_URL)

class StateManager:
    """
    Manages the storage and retrieval of intermediate task data in Redis.
    This prevents passing large data blobs (like images) between Celery tasks,
    which is a critical best practice for distributed systems.
    """
    def __init__(self, request_id: str):
        """
        Initializes the manager with a unique ID for the current OCR request.

        Args:
            request_id (str): A unique identifier (e.g., a UUID) for the entire workflow.
        """
        if not request_id:
            raise ValueError("request_id cannot be empty.")
        self.request_id = request_id
        # Set a default TTL (Time-To-Live) of 2 hours for all keys related to this request.
        self.ttl_seconds = 7200

    def _get_key(self, key: str) -> str:
        """Constructs a unique, namespaced Redis key for the current request."""
        return f"ocr_state:{self.request_id}:{key}"

    def _set_data(self, key: str, data: Any):
        """Serializes data using pickle and stores it in Redis with a timeout."""
        redis_key = self._get_key(key)
        redis_client.set(redis_key, pickle.dumps(data), ex=self.ttl_seconds)

    def _delete_key(self, key: str):
        """Removes a Redis key if it exists."""
        redis_key = self._get_key(key)
        redis_client.delete(redis_key)

    def _get_data(self, key: str) -> Any:
        """Retrieves and deserializes data from Redis."""
        redis_key = self._get_key(key)
        serialized_data = redis_client.get(redis_key)
        if serialized_data is None:
            raise KeyError(f"Data for key '{key}' not found for request_id '{self.request_id}'. The key may have expired or was never set.")
        return pickle.loads(serialized_data)

    # --- Public methods for managing specific workflow data ---

    def save_initial_images(self, images: List[np.ndarray]):
        """
        Saves the decoded page images. Each page is stored under its own Redis key
        so we do not push massive blobs through a single SET command.
        """
        try:
            existing_indices = self.load_page_indices()
        except KeyError:
            existing_indices = []

        for index in existing_indices:
            self._delete_key(f"initial_image:{index}")

        for index, image in enumerate(images):
            payload = self._encode_page_image(image)
            self._set_data(f"initial_image:{index}", payload)

        self._set_data("page_indices", list(range(len(images))))

    def load_page_image(self, page_index: int) -> np.ndarray:
        """
        Loads a single page image. Prefer the per-page key, but fall back to the
        legacy list-based storage if it still exists for older tasks.
        """
        try:
            stored_image = self._get_data(f"initial_image:{page_index}")
            return self._decode_page_image(stored_image)
        except KeyError:
            images = self._get_data("initial_images")
            if page_index >= len(images):
                raise IndexError("Page index out of range.")
            return images[page_index]

    def save_line_boxes(self, page_index: int, boxes: list):
        """Stores detected line boxes for a page."""
        self._set_data(f"line_boxes:{page_index}", boxes)

    def load_line_boxes(self, page_index: int) -> list:
        """Retrieves detected line boxes for a page."""
        return self._get_data(f"line_boxes:{page_index}")

    def save_word_polygons(self, page_index: int, polygons: list):
        """Stores detected word polygons for a page."""
        self._set_data(f"word_polygons:{page_index}", polygons)

    def load_word_polygons(self, page_index: int) -> list:
        """Retrieves detected word polygons for a page."""
        return self._get_data(f"word_polygons:{page_index}")

    def load_page_indices(self) -> List[int]:
        """Returns the list of page indices for this request."""
        return self._get_data("page_indices")

    def save_page_result(self, page_index: int, text: str, confidence: float):
        """Saves the final OCR result for a single page."""
        page_result = {
            "page_index": page_index,
            "text": text,
            "confidence": confidence
        }
        self._set_data(f"page_result:{page_index}", page_result)

    def load_all_page_results(self, page_indices: List[int]) -> List[dict]:
        """Loads and returns all specified page results, sorted by page index."""
        results = []
        for i in sorted(page_indices):
            results.append(self._get_data(f"page_result:{i}"))
        return results

    def _encode_page_image(self, image: np.ndarray) -> Any:
        """
        Compress the image to PNG before storing it in Redis. This keeps payloads
        small and consistent regardless of the original array size.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected a numpy.ndarray for image storage.")

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image to PNG for Redis storage.")
        return {"storage": "png", "data": encoded.tobytes()}

    def _decode_page_image(self, payload: Any) -> np.ndarray:
        """
        Reconstructs the numpy image from the stored payload, handling both the
        compressed and legacy raw-array formats.
        """
        if isinstance(payload, dict) and payload.get("storage") == "png":
            encoded_bytes = payload.get("data", b"")
            if not isinstance(encoded_bytes, (bytes, bytearray)):
                raise TypeError("Invalid PNG payload stored in Redis.")
            array = np.frombuffer(encoded_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode PNG image retrieved from Redis.")
            return image

        if isinstance(payload, np.ndarray):
            return payload

        raise TypeError("Unsupported image payload type retrieved from Redis.")

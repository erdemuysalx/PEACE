"""Depth projection service — projects 2D detections to 3D world coordinates."""

import math
import numpy as np

from peace.schemas import Object, ObjectWithDepth


class DepthProjectionService:
    """
    Projects detected objects from image space to 3D world coordinates.

    Uses the pinhole camera model and a depth image to calculate the distance
    to each detected object, then transforms to the world (ENU) frame using
    the drone's current pose.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Store camera intrinsics and depth filtering parameters."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold

    def update_intrinsics(
        self, fx: float, fy: float, cx: float, cy: float, width: int, height: int
    ) -> None:
        """Update camera intrinsics (called when CameraInfo arrives)."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    def project_detections(
        self,
        objects: list[Object],
        depth_img: np.ndarray,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        heading: float,
    ) -> list[ObjectWithDepth]:
        """
        Calculate depth and world position for a list of detected objects.

        Args:
            objects: Detected objects with bounding boxes.
            depth_img: Depth image in metres (float32).
            pos_x, pos_y, pos_z: Drone position in local ENU frame.
            heading: Drone yaw in radians.

        Returns:
            List of ObjectWithDepth (one per input object, in the same order).
        """
        result: list[ObjectWithDepth] = []

        for obj in objects:
            base = obj.model_dump()

            conf = obj.confidence / 100.0
            if conf < self.confidence_threshold:
                result.append(ObjectWithDepth(**base, depth_meters=None, world_pos=[0.0, 0.0, 0.0], depth_valid=False))
                continue

            bbox = obj.bbox
            if len(bbox) != 4:
                result.append(ObjectWithDepth(**base, depth_meters=None, world_pos=[0.0, 0.0, 0.0], depth_valid=False))
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, self.width - 1))
            x2 = max(0, min(x2, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            y2 = max(0, min(y2, self.height - 1))

            if x2 <= x1 or y2 <= y1:
                result.append(ObjectWithDepth(**base, depth_meters=None, world_pos=[0.0, 0.0, 0.0], depth_valid=False))
                continue

            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2

            # Sample a small patch around the bbox centre for robustness
            patch = depth_img[
                max(0, cy_px - 5) : min(self.height, cy_px + 20),
                max(0, cx_px - 5) : min(self.width, cx_px + 20),
            ]
            valid = patch[(patch > self.min_depth) & (patch < self.max_depth)]

            if len(valid) == 0:
                result.append(ObjectWithDepth(**base, depth_meters=None, world_pos=[0.0, 0.0, 0.0], depth_valid=False))
                continue

            obj_depth = float(np.median(valid))
            angle = self._pixel_to_angle(cx_px)
            world_x, world_y = self._to_world(obj_depth, angle, pos_x, pos_y, heading)

            result.append(
                ObjectWithDepth(
                    **base,
                    depth_meters=obj_depth,
                    world_pos=[world_x, world_y, pos_z],
                    depth_valid=True,
                )
            )

        return result

    # ── Internal geometry helpers ────────────────────────────────────────────

    def _pixel_to_angle(self, pixel_x: int) -> float:
        """Convert pixel x-coordinate to horizontal angle using pinhole model."""
        return math.atan2(pixel_x - self.cx, self.fx)

    def _to_world(
        self, depth: float, angle: float, pos_x: float, pos_y: float, heading: float
    ) -> tuple[float, float]:
        """Transform camera-frame depth + angle to world ENU coordinates."""
        rel_x = depth * math.cos(angle)
        rel_y = depth * math.sin(angle)
        world_x = pos_x + rel_x * math.cos(heading) - rel_y * math.sin(heading)
        world_y = pos_y + rel_x * math.sin(heading) + rel_y * math.cos(heading)
        return world_x, world_y

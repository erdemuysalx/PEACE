"""Vision node — thin ROS 2 wrapper around ObjectDetector and DepthProjectionService."""

import concurrent.futures
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from peace.backends.detector import ObjectDetector, VlmDetector, YoloDetector
from peace.core.config import get_settings
from peace.system_prompts import get_perception_prompt
from peace.schemas import SceneDescription, ObjectWithDepth
from peace.services.depth_service import DepthProjectionService

# Default camera intrinsics — overridden from /camera/color/camera_info
_DEFAULT_FX = 1397.2236
_DEFAULT_FY = 1397.2236
_DEFAULT_CX = 960.0
_DEFAULT_CY = 540.0
_DEFAULT_WIDTH = 1920
_DEFAULT_HEIGHT = 1080


class VisionNode(Node):
    """
    Perception node that converts camera frames into 3D object detections.

    Subscribes to:
      - /camera/color/image        — RGB frames for detection
      - /camera/depth/image        — depth frames for range estimation
      - /camera/color/camera_info  — intrinsics for the pinhole projection model
      - /mavros/local_position/odom — drone pose for ENU world-frame projection

    Publishes:
      - /agent/vision/detections   — JSON-serialised SceneDescription with world positions
      - /agent/vision/annotated_img — BGR image with bounding-box overlays

    A periodic timer fires at vision_inference_rate Hz and submits one inference
    job at a time to a single-worker thread pool, so slow backends (VLM) never
    block ROS callbacks. The detection backend (YOLO or VLM) is chosen by
    settings.use_yolo and must satisfy the ObjectDetector protocol.
    """

    def __init__(self, detector: Optional[ObjectDetector] = None) -> None:
        """Set up detector, depth service, camera/odometry subscribers, and publishers."""
        super().__init__("vision_node")
        self.settings = get_settings()

        # ── Image states ─────────────────────────────────────────────────────
        self._color_img: Optional[np.ndarray] = None
        self._depth_img: Optional[np.ndarray] = None
        self._img_lock = threading.Lock()

        # ── Position state ───────────────────────────────────────────────────
        self._pos_x = self._pos_y = self._pos_z = self._heading = 0.0
        self._pos_lock = threading.Lock()

        # ── Single-slot thread pool (one inference running at a time) ────────
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._inference_running = False
        
        # ── Vision backend ───────────────────────────────────────────────────
        if detector is None:
            detector = self._build_detector()
        self._detector = detector

        # ── Services ─────────────────────────────────────────────────────────
        self._depth_svc = DepthProjectionService(
            fx=_DEFAULT_FX,
            fy=_DEFAULT_FY,
            cx=_DEFAULT_CX,
            cy=_DEFAULT_CY,
            width=_DEFAULT_WIDTH,
            height=_DEFAULT_HEIGHT,
            min_depth=self.settings.min_depth,
            max_depth=self.settings.max_depth,
            confidence_threshold=self.settings.vision_confidence_threshold,
        )

        self._bridge = CvBridge()

        # ── QoS profiles ─────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Callback groups ──────────────────────────────────────────────────
        reentrant_cb = ReentrantCallbackGroup()
        mutex_cb = MutuallyExclusiveCallbackGroup()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo,
            self.settings.topic_camera_info,
            self._camera_info_cb,
            reliable_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            Image,
            self.settings.topic_camera_color_image,
            self._color_img_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            Image,
            self.settings.topic_camera_depth_image,
            self._depth_img_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            Odometry,
            self.settings.topic_odom,
            self._odom_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._detections_pub = self.create_publisher(
            String, "/agent/vision/detections", reliable_qos
        )
        self._annotated_pub = self.create_publisher(
            Image, "/agent/vision/annotated_img", best_effort_qos
        )

        # Periodic inference timer
        inference_period = 1.0 / self.settings.vision_inference_rate
        self.create_timer(
            inference_period, self._inference_cb, callback_group=mutex_cb
        )

        backend = "YOLO" if self.settings.use_yolo else "VLM"
        self.get_logger().info(
            f"Vision Node ready | backend={backend} | "
            f"rate={self.settings.vision_inference_rate}Hz | "
            f"confidence={self.settings.vision_confidence_threshold}"
        )

    # ── Detector factory ─────────────────────────────────────────────────────

    def _build_detector(self) -> ObjectDetector:
        """Instantiate and return the detector selected by settings (YOLO or VLM)."""
        if self.settings.use_yolo:
            det = YoloDetector(
                model_path=self.settings.yolo_model,
                confidence_threshold=self.settings.yolo_confidence_threshold,
                device=self.settings.yolo_device,
            )
            if not det.is_ready:
                self.get_logger().error(f"YOLO init failed: {det.error_message}")
            return det
        else:
            from ollama import Client

            client = Client(host=self.settings.ollama_endpoint)
            det = VlmDetector(
                client=client,
                model=self.settings.vision_model,
                system_prompt=get_perception_prompt(),
                temperature=self.settings.vision_model_temperature,
                num_predict=self.settings.vision_model_num_predict,
                confidence_threshold=self.settings.vision_confidence_threshold,
            )
            if not det.is_ready:
                self.get_logger().error(
                    f"VLM model '{self.settings.vision_model}' not available at "
                    f"{self.settings.ollama_endpoint}. "
                    f"Run: ollama pull {self.settings.vision_model}"
                )
            return det

    # ── Camera and odometry callbacks ─────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Update depth service with camera intrinsics from CameraInfo."""
        K = msg.k
        self._depth_svc.update_intrinsics(
            fx=K[0], fy=K[4], cx=K[2], cy=K[5],
            width=msg.width, height=msg.height,
        )

    def _color_img_cb(self, msg: Image) -> None:
        """Convert and store the latest BGR color image, resizing to intrinsic resolution."""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            w, h = self._depth_svc.width, self._depth_svc.height
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            with self._img_lock:
                self._color_img = img
        except Exception as e:
            self.get_logger().error(f"Color image error: {e}")

    def _depth_img_cb(self, msg: Image) -> None:
        """Convert and store the latest depth image in metres (float32)."""
        try:
            if msg.encoding == "32FC1":
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                scale = 1.0
            elif msg.encoding == "16UC1":
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                scale = 0.001
            else:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                scale = 0.001

            w, h = self._depth_svc.width, self._depth_svc.height
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

            with self._img_lock:
                self._depth_img = img.astype(np.float32) * scale
        except Exception as e:
            self.get_logger().error(f"Depth image error: {e}")

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache current drone position and heading for depth-to-world projection."""
        quat = msg.pose.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        heading = math.atan2(siny_cosp, cosy_cosp)

        with self._pos_lock:
            self._pos_x = msg.pose.pose.position.x
            self._pos_y = msg.pose.pose.position.y
            self._pos_z = msg.pose.pose.position.z
            self._heading = heading

    # ── Periodic inference timer ───────────────────────────────────────────────────────

    def _inference_cb(self) -> None:
        """Timer callback — submit an inference job if images are available and none is running."""
        if self._inference_running:
            return
        if not self._detector.is_ready:
            self.get_logger().warn("Detector not ready", throttle_duration_sec=10.0)
            return

        with self._img_lock:
            if self._color_img is None or self._depth_img is None:
                self.get_logger().warn("Images not available", throttle_duration_sec=5.0)
                return
            color_img, depth_img = self._color_img.copy(), self._depth_img.copy()

        with self._pos_lock:
            pos_x, pos_y, pos_z, heading = (
                self._pos_x, self._pos_y, self._pos_z, self._heading
            )

        self._inference_running = True
        self._executor.submit(
            self._run_inference, color_img, depth_img, pos_x, pos_y, pos_z, heading
        )

    # ── Inference handling ────────────────────────────────────────────────────────

    def _run_inference(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        heading: float,
    ) -> None:
        """Run detection, project to 3D, and publish annotated results."""
        try:
            start = time.time()
            objects = self._depth_svc.project_detections(
                self._detector.detect(color_img), depth_img, pos_x, pos_y, pos_z, heading
            )

            valid = [o for o in objects if o.depth_valid]

            image_desc = SceneDescription(
                summary=f"{len(valid)} object(s) with valid depth",
                objects=objects,
                camera_pos=[pos_x, pos_y, pos_z],
                img_res=[color_img.shape[1], color_img.shape[0]],
            )

            det_msg = String()
            det_msg.data = image_desc.model_dump_json()
            self._detections_pub.publish(det_msg)

            if not valid:
                self.get_logger().debug(
                    f"No objects with valid depth ({time.time() - start:.2f}s)"
                )
                return

            annotated = self._annotate(color_img, objects)
            self._annotated_pub.publish(
                self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            )

            if self.settings.debug_save_images:
                self._save_debug_image(annotated)

            elapsed = time.time() - start
            if len(valid) == 1:
                obj = valid[0]
                self.get_logger().info(
                    f"{obj.label} @ ({obj.world_pos[0]:.1f}, {obj.world_pos[1]:.1f})m "
                    f"conf={obj.confidence / 100:.0%} ({elapsed:.2f}s)"
                )
            else:
                summary = ", ".join(
                    f"{o.label}@({o.world_pos[0]:.1f},{o.world_pos[1]:.1f})"
                    for o in valid
                )
                self.get_logger().info(
                    f"{len(valid)}/{len(objects)} objects: {summary} ({elapsed:.2f}s)"
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {type(e).__name__}: {e}")
        finally:
            self._inference_running = False

    # ── Image annotation ─────────────────────────────────────────────────────

    def _annotate(self, img: np.ndarray, objects: list[ObjectWithDepth]) -> np.ndarray:
        """Draw bounding boxes and depth labels onto a copy of the image."""
        out = img.copy()
        for obj in objects:
            conf = obj.confidence / 100.0
            color = (
                (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 165, 255)
            )
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if obj.depth_valid and obj.depth_meters is not None:
                label = f"{obj.label}: {obj.depth_meters:.1f}m ({conf:.0%})"
            else:
                label = f"{obj.label} (no depth)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return out

    def _save_debug_image(self, img: np.ndarray) -> None:
        """Write the annotated image to the debug output directory with a timestamp filename."""
        import os
        self.get_logger().debug(f"Saving debug image")        
        out_dir = self.settings.debug_output_dir
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"{time.time():.3f}.jpg")
        cv2.imwrite(filename, img)


def main(args=None) -> None:
    """Initialise ROS2, spin the VisionNode, and shut down on exit."""
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
板端推理适配器.
"""
from typing import Dict
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class EdgeInferenceAdapter:
    def __init__(self, model_path: str, backend: str = "rknn", input_size: tuple = (640, 640)):
        self.backend = backend.lower()
        self.model_path = model_path
        self.input_size = input_size

        if self.backend == "rknn":
            self._init_rknn()
        elif self.backend == "ascend":
            self._init_ascend()
        elif self.backend == "onnx":
            self._init_onnx()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_rknn(self):
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {self.model_path}")
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError("Failed to initialize RKNN runtime")
            logger.info("RKNN runtime initialized successfully.")
        except ImportError:
            logger.error("rknnlite not installed.")
            raise

    def _init_ascend(self):
        try:
            import acl
            ret = acl.init()
            if ret != 0:
                raise RuntimeError("Failed to initialize ACL")
            logger.info("Ascend ACL initialized successfully.")
        except ImportError:
            logger.warning("Ascend ACL not available.")

    def _init_onnx(self):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        logger.info(f"ONNX session created with providers: {providers}")

    def infer(self, image: np.ndarray) -> Dict:
        input_tensor = self._preprocess(image)
        if self.backend == "rknn":
            outputs = self.rknn.inference(inputs=[input_tensor])
            return {"raw_output": outputs}
        elif self.backend == "onnx":
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            return {"raw_output": outputs}
        else:
            raise NotImplementedError

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        h, w = self.input_size
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def release(self):
        if self.backend == "rknn" and hasattr(self, "rknn"):
            self.rknn.release()
        logger.info("Edge inference resources released.")

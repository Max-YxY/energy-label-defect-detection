"""Inference module for PC and edge deployment."""
from .detector import EnergyLabelDetector
from .postprocess import PostProcessor

__all__ = ["EnergyLabelDetector", "PostProcessor"]

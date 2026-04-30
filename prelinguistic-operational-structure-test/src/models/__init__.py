from .base import BasePLOSModel
from .field_model import FieldModel
from .flow_checkpoint_model import FlowCheckpointModel
from .koopman_model import KoopmanModel
from .patch_graph_model import PatchGraphModel
from .pixel_predictor import PixelPredictor
from .predictive_coding_model import PredictiveCodingModel
from .schema_model import SchemaModel
from .slot_model import SlotModel
from .trajectory_memory import TrajectoryMemory
from .world_model import WorldModel


def make_model(name: str) -> BasePLOSModel:
    if name == "pixel_predictor":
        return PixelPredictor()
    if name == "predictive_coding_model":
        return PredictiveCodingModel()
    if name == "patch_graph_model":
        return PatchGraphModel()
    if name == "koopman_model":
        return KoopmanModel()
    if name == "trajectory_memory":
        return TrajectoryMemory()
    if name == "world_model":
        return WorldModel()
    if name == "slot_model":
        return SlotModel()
    if name == "field_model":
        return FieldModel()
    if name == "flow_checkpoint_model":
        return FlowCheckpointModel()
    if name == "schema_model":
        return SchemaModel()
    raise ValueError(f"unknown model: {name}")

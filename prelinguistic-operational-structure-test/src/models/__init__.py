from .base import BasePLOSModel
from .field_memory_model import FieldMemoryModel
from .field_model import FieldModel
from .flow_checkpoint_model import FlowCheckpointModel
from .koopman_model import KoopmanModel
from .patch_graph_model import PatchGraphModel
from .pixel_predictor import PixelPredictor
from .predictive_coding_model import PredictiveCodingModel
from .recurrent_flow_checkpoint_model import RecurrentFlowCheckpointModel
from .schema_model import SchemaModel
from .schema_memory_model import SchemaMemoryModel
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
    if name == "recurrent_flow_checkpoint_model":
        return RecurrentFlowCheckpointModel()
    if name == "field_memory_model":
        return FieldMemoryModel()
    if name == "schema_memory_model":
        return SchemaMemoryModel()
    if name in {
        "recurrent_flow_checkpoint_no_shared_selector",
        "field_memory_no_shared_selector",
        "schema_memory_no_shared_selector",
    }:
        from ..b22_selector_free_models import make_selector_free_model

        return make_selector_free_model(name)
    if name == "schema_model":
        return SchemaModel()
    raise ValueError(f"unknown model: {name}")

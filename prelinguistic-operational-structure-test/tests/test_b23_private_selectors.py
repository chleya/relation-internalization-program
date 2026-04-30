from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b23_private_selectors import validate_no_shared_selector_call
from src.model_io import make_model_batch
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_private_selector_models_report_private_provenance():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 0, delay=4), config)
    for name in ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"]:
        model = make_model(name)
        output = model.forward(batch)
        structure = output["structure"]
        assert structure["selected_region"] >= 0
        assert structure["shared_selector_used"] is False
        assert structure["model_private_score_used"] is True
        assert isinstance(structure["fallback_used"], bool)


def test_validate_no_shared_selector_call_passes():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 1, delay=4), config)
    result = validate_no_shared_selector_call(make_model("field_memory_model"), batch)
    assert result["shared_selector_used"] is False
    assert result["pass"] == 1

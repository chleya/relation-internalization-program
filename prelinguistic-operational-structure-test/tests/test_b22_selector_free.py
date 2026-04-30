from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b22_selector_free_models import make_selector_free_model
from src.inspect_policy import select_region_from_logits
from src.model_io import make_model_batch
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_selector_free_variants_instantiate_from_factory():
    for name in [
        "recurrent_flow_checkpoint_no_shared_selector",
        "field_memory_no_shared_selector",
        "schema_memory_no_shared_selector",
    ]:
        model = make_model(name)
        assert model.name == name


def test_selector_free_variant_reports_no_shared_selector():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 2, delay=4)
    batch = make_model_batch(episode, config)
    model = make_selector_free_model("recurrent_flow_checkpoint_model")
    output = model.forward(batch)
    assert select_region_from_logits(output["inspection_logits"]) >= 0
    assert output["structure"]["shared_selector_used"] is False
    assert output["structure"]["model_private_score_used"] is True

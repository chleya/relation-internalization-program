from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b23_selector_validation import ablate_model_private_trace, ablate_shared_selector_path, evaluate_b23_source_ablation
from src.model_io import make_model_batch
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_private_and_shared_ablation_do_not_crash():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 0, delay=4), config)
    model = make_model("field_memory_model")
    private = ablate_model_private_trace(model, batch)
    shared = ablate_shared_selector_path(model, batch)
    assert private["applicable"] is True
    assert shared["applicable"] is True


def test_source_ablation_metrics_nonnegative():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    metrics = evaluate_b23_source_ablation(make_model("schema_memory_model"), [episode], config)
    assert metrics["model_private_trace_drop"] >= 0.0
    assert metrics["shared_selector_ablation_drop"] >= 0.0

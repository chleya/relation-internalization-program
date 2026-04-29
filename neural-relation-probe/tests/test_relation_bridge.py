from src.data import make_dataset
from src.model import hidden_states, train_mlp
from src.probe import train_probe
from src.relation_bridge import extract_relation_table, relation_alignment, table_resource_accuracy
from src.run_extract import gated_extraction_score


def test_extract_relation_table_runs_and_edits():
    train = make_dataset(400, seed=0, mode="base")
    test = make_dataset(100, seed=1, mode="ood")
    model = train_mlp(train.x, train.y_resource, seed=0)
    probe = train_probe(hidden_states(model, train.x), train.y_relation, seed=0)
    table = extract_relation_table(model, probe)

    assert len(table.describe_relations()) == 6
    assert relation_alignment(table) == 1.0
    assert table_resource_accuracy(table, test.contexts, test.y_resource) > 0.8

    context = {"texture": "A", "wet": "dry", "color": "blue", "odor": "weak"}
    assert table.act(context) == "eat"
    assert table.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    assert table.act(context) == "avoid"


def test_gated_extraction_score_rejects_failed_gate():
    passing = {
        "table_ood_accuracy": 1.0,
        "table_spurious_attack_accuracy": 1.0,
        "table_relation_alignment": 1.0,
        "edit_success": True,
    }
    failing = dict(passing)
    failing["table_spurious_attack_accuracy"] = 0.5

    assert gated_extraction_score(passing) > 0.0
    assert gated_extraction_score(failing) == 0.0

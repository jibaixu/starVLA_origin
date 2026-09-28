from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBODOJO_CONFIG = (
    ROOT
    / "examples"
    / "simBenchmarks"
    / "RoboDojo"
    / "train_files"
    / "starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml"
)


def test_robodojo_recipe_selects_qwenpi_v3_and_canonical_attention():
    config = ROBODOJO_CONFIG.read_text(encoding="utf-8")

    assert "name: QwenPI_v3" in config
    assert "action_model_type: LayerwiseFM" in config
    assert "action_dit_hidden_dim: 1024" in config
    assert "interleave_self_attention: true" in config
    assert "action_dim: 14" in config
    assert "state_dim: 14" in config
    assert "action_horizon: 50" in config


def test_qwenpi_v3_defaults_to_legacy_mode_but_supports_alias():
    framework = (ROOT / "starVLA/model/framework/VLM4A/QwenPI_v3.py").read_text(encoding="utf-8")
    dit = (ROOT / "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py").read_text(
        encoding="utf-8"
    )

    assert '"interleave_self_attention": False' in framework
    assert "use_canonical_forward: Optional[bool] = None" in dit
    assert "self._interleave_self_attention = bool(interleave_self_attention)" in dit

import json
from types import SimpleNamespace

from starVLA.training.train_starvla import VLATrainer


class _IteratorLoader:
    def __init__(self):
        self.epochs = []

    def __len__(self):
        return 3

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        return iter(("first", "second", "third"))


def test_data_iterator_does_not_replay_saved_cursor():
    trainer = object.__new__(VLATrainer)
    trainer.vla_train_dataloader = _IteratorLoader()
    trainer.vla_epoch_count = 0
    trainer.vla_batches_in_epoch = 10_100
    trainer._data_iterators_initialized = False

    trainer._create_data_iterators()

    assert next(trainer.vla_iter) == "first"
    assert trainer.vla_train_dataloader.epochs == [0]


def test_light_resume_ignores_legacy_data_cursor(tmp_path):
    trainer = object.__new__(VLATrainer)
    trainer.resume_progress_path = str(tmp_path / "latest_resume.json")
    trainer.completed_steps = 10_000
    trainer.vla_batches_consumed = 1
    trainer.vla_epoch_count = 2
    trainer.vla_batches_in_epoch = 37
    (tmp_path / "latest_resume.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "completed_steps": 10_000,
                "vla_batches_consumed": 10_100,
                "vla_epoch_count": 4,
                "vla_batches_in_epoch": 99,
            }
        ),
        encoding="utf-8",
    )

    trainer._load_light_resume_progress()

    assert (trainer.vla_batches_consumed, trainer.vla_epoch_count, trainer.vla_batches_in_epoch) == (0, 0, 0)


def test_adaptive_sidecar_path_for_portable_checkpoint():
    trainer = object.__new__(VLATrainer)

    assert trainer._adaptive_state_path("/tmp/steps_10000_pytorch_model.pt") == "/tmp/steps_10000_adaptive.json"
    assert trainer._adaptive_state_path("/tmp/steps_10000_model.safetensors") == "/tmp/steps_10000_adaptive.json"


def test_adaptive_sidecar_is_restored(tmp_path):
    class _AdaptiveModel:
        def __init__(self):
            self.state = None

        def set_adaptive_state(self, state):
            self.state = state

    class _Accelerator:
        def __init__(self, model):
            self.model = model

        def unwrap_model(self, model):
            return self.model

    trainer = object.__new__(VLATrainer)
    model = _AdaptiveModel()
    trainer.model = model
    trainer.accelerator = _Accelerator(model)
    checkpoint = tmp_path / "steps_10000_pytorch_model.pt"
    checkpoint.write_bytes(b"placeholder")
    (tmp_path / "steps_10000_adaptive.json").write_text(
        json.dumps({"mode": "dwa", "weights": {"vla": 1.25}}),
        encoding="utf-8",
    )

    trainer._load_adaptive_state(str(checkpoint))

    assert model.state == {"mode": "dwa", "weights": {"vla": 1.25}}


def test_checkpoint_keeps_existing_data_iterator(tmp_path):
    class _Accelerator:
        is_main_process = True

        def get_state_dict(self, model):
            return {}

        def unwrap_model(self, model):
            return model

        def print(self, message):
            self.message = message

        def wait_for_everyone(self):
            pass

    trainer = object.__new__(VLATrainer)
    trainer.accelerator = _Accelerator()
    trainer.config = SimpleNamespace(
        output_dir=str(tmp_path),
        trainer=SimpleNamespace(save_format="pt"),
    )
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    trainer.resume_progress_path = str(tmp_path / "checkpoints" / "latest_resume.json")
    trainer.completed_steps = 10_000
    trainer.vla_iter = object()
    trainer.model = object()
    (tmp_path / "checkpoints").mkdir()

    trainer._save_checkpoint()

    assert trainer.vla_iter is not None
    assert (tmp_path / "checkpoints" / "steps_10000_pytorch_model.pt").is_file()

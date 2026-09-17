from starVLA.training.train_starvla import TrainingProgress


def test_training_progress_round_trip_preserves_data_cursor():
    progress = TrainingProgress(
        completed_steps=10_000,
        vla_batches_consumed=10_100,
        vla_epoch_count=2,
        vla_batches_in_epoch=37,
        adaptive_state={"loss_weights": {"vla": 1.0}},
    )

    restored = TrainingProgress()
    restored.load_state_dict(progress.state_dict())

    assert restored == progress

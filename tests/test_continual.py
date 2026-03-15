from cjepa import ContinualExperimentConfig, build_pretrain_command, write_dataset_csv


def test_build_pretrain_command_contains_expected_flags() -> None:
    config = ContinualExperimentConfig(task_name="task2_adapt")
    command = build_pretrain_command(config)
    assert command[:3] == ["python", "-m", "app.main"]
    assert "--tag" in command
    assert command[-1] == "task2_adapt"


def test_write_dataset_csv(tmp_path) -> None:
    csv_path = write_dataset_csv(
        entries=[("/tmp/a.mp4", 0), ("/tmp/b.mp4", 1)],
        destination=tmp_path / "train.csv",
    )
    assert csv_path.read_text(encoding="utf-8") == "/tmp/a.mp4 0\n/tmp/b.mp4 1\n"

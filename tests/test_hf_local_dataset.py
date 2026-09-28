"""Optional check against a locally generated pipeline output."""

from pathlib import Path

import pytest

from world_model.data import HFWindowDataset


@pytest.mark.local_data
def test_pipeline_output_loads_sampled_window(pytestconfig):
    output = Path(pytestconfig.getoption("--pipeline-out"))
    candidates = [
        output / "selected" / "hf_dataset",
        output / "episodes" / "hf_dataset",
    ]
    dataset_path = next(
        (path for path in candidates if (path / "state.json").is_file()), None
    )
    assert dataset_path is not None, (
        f"No saved Hugging Face dataset under {output}; expected selected/hf_dataset "
        "or episodes/hf_dataset. Run the dataset pipeline first."
    )

    dataset = HFWindowDataset(
        str(dataset_path), img_size=32, sample_mode=True, sample_k=4
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["rgb"].shape == (4, 3, 32, 32)
    assert sample["odom"].shape == (4, dataset.D_odom)

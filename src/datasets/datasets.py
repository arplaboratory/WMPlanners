from .videodataset import VideoDataset
from .pairdataset import PairDataset
from .datasample import VideoSample
from .utils.collate import video_collate, pair_collate
from torch.utils.data import DataLoader
from typing import Optional


def build_dataloader(
    video_paths: list[str],
    labels: Optional[list[int]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    datasetargs: Optional[dict] = None,
    type:str = 'pair',
) -> DataLoader:
    
    if labels is None:
        samples = [VideoSample(video_path=p, label=None) for p in video_paths]
    else:
        if len(video_paths) != len(labels):
            raise ValueError("video_paths and labels must have the same length.")
        samples = [VideoSample(video_path=p, label=y) for p, y in zip(video_paths, labels)]

    collate_fn = None
    if type == 'video':

        dataset = VideoDataset(
            samples=samples,
            num_frames=datasetargs.get('num_frames', 16),
            sampling_stride=datasetargs.get('sampling_stride', 4),
            crop_size=datasetargs.get('crop_size', 256),
            random_clip=datasetargs.get('random_clip', True),
        )
        colate_fn = video_collate
    elif type == 'pair':
        dataset = PairDataset(
            samples=samples,
            dataset=PairDataset(
                samples=samples,
                random_pair=datasetargs.get('random_pair', True),
                image_size=datasetargs.get('image_size', 256),
            )
        )
        collate_fn = pair_collate

        

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=datasetargs.get('random_pair', False) or datasetargs.get('random_clip', False),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
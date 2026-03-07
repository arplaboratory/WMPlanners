import torch

def video_collate(batch: list[dict]) -> dict:
    '''
    Collate function for video datasets. Expects each item in the batch to be a dict with keys:
      - "video": Tensor of shape [C, T, H, W]
      - "label": Optional int
      - "path": str
      - "frame_indices": Tensor of shape [T]
      '''
    videos = torch.stack([x["video"] for x in batch], dim=0)  # [B, C, T, H, W]
    labels = [x["label"] for x in batch]
    paths = [x["path"] for x in batch]
    frame_indices = [x["frame_indices"] for x in batch]

    if all(label is not None for label in labels):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None

    return {
        "video": videos,
        "label": labels,
        "path": paths,
        "frame_indices": frame_indices,
    }

def pair_collate(batch: list[dict]) -> dict:
    '''
    Collate function for paired frame datasets. Expects each item in the batch to be a dict with keys:
      - "frame_t": Tensor of shape [C, H, W]
      - "frame_t1": Tensor of shape [C, H, W]
      - "label": Optional int
      - "path": str
      - "indices": Tensor of shape [2] (the frame indices)
      '''
    frame_t = torch.stack([x["frame_t"] for x in batch], dim=0)    # [B, C, H, W]
    frame_t1 = torch.stack([x["frame_t1"] for x in batch], dim=0)  # [B, C, H, W]

    labels_list = [x["label"] for x in batch]
    paths = [x["path"] for x in batch]
    indices = torch.stack([x["indices"] for x in batch], dim=0)    # [B, 2]

    labels = None
    if all(label is not None for label in labels_list):
        labels = torch.tensor(labels_list, dtype=torch.long)

    return {
        "frame_t": frame_t,
        "frame_t1": frame_t1,
        "label": labels,
        "path": paths,
        "indices": indices,
    }
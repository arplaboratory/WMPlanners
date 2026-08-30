# Changelog

All notable changes to this project should be documented in this file.

## [1.2.0] - 2026-08-30
### Added
- Added a manifest-first dataset workflow with the `wm-dataset` adapter, a direct training loader, and lossless NPZ and Hugging Face exports that preserve dense depth and validity metadata.
- Added a unified launcher for standard and DINO-based world-model training, plus direct dataset training support.
- Added generic, Spot, and rover episode-selection profiles; Apollo robot configuration; Collect3 archive extraction; and a Rerun review-to-Parquet pipeline.
- Added tooling to update existing Apptainer sandboxes and optionally rebuild their SIF images.

### Changed
- Modularized the world-model training backends and expanded the shared dataset and batch interfaces.
- Improved episode generation with configurable image-gap detection, nearest-sensor synchronization checks, recursive MCAP input expansion, and richer curation documentation.
- Updated Hugging Face datasets to retain full depth frames, timestamps, split assignments, and per-sensor validity masks.

### Fixed
- Fixed Apptainer setup so configured sandbox paths and existing virtual environments are preserved.
- Fixed external dependency installation for source-free containers and added the missing ROS bag and MCAP runtime dependencies.

## [1.1.0] - 2026-05-02
### Added
- Updated World Model architectures and supporting training code and HPC scripts.

## [1.0.0] - 2026-03-15
### Added
- Added Dino Encoder Support, Data loaders for video stream and pairwise images. MVP for world model training.

## [0.1.0] - 2026-02-15
### Added
- Initial brain dump for DS GA 3001 project which includes porting over previous work in RL and python-based project templates.

# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-09-27

First public release of World Model Planners.

### Added
- Data pipeline for ROS 1 bags and ROS 2 MCAP recordings, including sensor alignment, fixed-length NPZ episodes, episode selection, and Hugging Face dataset export.
- `wm-dataset` tooling and loaders for NPZ episodes, Hugging Face datasets, and local Parquet tables.
- Training and evaluation paths for CNN, DINOv2, and V-JEPA2 world models, JEPA-lite, and DinoWM, with MLP or RSSM dynamics where supported.
- Cross-entropy method planning over learned model rollouts, with goal and risk scoring.
- ROS 2 planner bridge for standard world models and DinoWM, including recorded-bag replay and planned-trajectory publishing.
- Setup scripts, examples, and guides for data preparation, training, evaluation, and ROS 2 integration.

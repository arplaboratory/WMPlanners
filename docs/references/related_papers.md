# Related Papers: World Model for Dual-Mode Drone-Car Robot (race16)

Compiled April 2026. Papers organized by category with relevance to the race16 project.

---

## 1. World Models for Robotics (Core References)

### 1.1 PlaNet: Learning Latent Dynamics for Planning from Pixels
- **Authors:** Hafner et al.
- **Year/Venue:** ICML 2019
- **Link:** https://arxiv.org/abs/1811.04551
- **Summary:** Introduces the Recurrent State-Space Model (RSSM) with deterministic + stochastic latent states and multi-step latent overshooting objective. Performs MPC planning entirely in latent space. This is the foundational architecture behind your RSSM dynamics option and latent-space MPC planning.
- **Key Contribution:** RSSM architecture; latent-space CEM planning from pixels.

### 1.2 DreamerV2: Mastering Atari with Discrete World Models
- **Authors:** Hafner et al.
- **Year/Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2010.02193
- **Summary:** Extends PlaNet with discrete categorical latent representations and straight-through gradient training. Achieves human-level Atari performance via imagined rollouts. Directly relevant to the categorical RSSM variant and multi-step rollout loss design.
- **Key Contribution:** Discrete latent world models; actor-critic learning in imagination.

### 1.3 DreamerV3: Mastering Diverse Domains through World Models
- **Authors:** Hafner et al.
- **Year/Venue:** ICLR 2023 / Nature 2025
- **Link:** https://arxiv.org/abs/2301.04104
- **Summary:** General-purpose MBRL with a single set of hyperparameters across 150+ tasks. Introduces symlog normalization and other stabilization tricks for robust training. Directly relevant as the RSSM dynamics backbone and multi-step rollout training methodology for race16.
- **Key Contribution:** Robust, general-purpose world model with normalization tricks; first algorithm to collect diamonds in Minecraft from scratch.

### 1.4 DayDreamer: World Models for Physical Robot Learning
- **Authors:** Wu, Escontrela, Hafner, Goldberg, Abbeel
- **Year/Venue:** CoRL 2022
- **Link:** https://arxiv.org/abs/2206.14176
- **Summary:** Deploys Dreamer on physical robots (quadruped, robot arm) by learning world models from real-world multi-modal sensory data (vision + proprioception). Directly relevant as a precedent for fusing visual and proprioceptive inputs into a latent world model on a physical robot.
- **Key Contribution:** First demonstration of Dreamer-style world models on real physical robots with multimodal sensors.

### 1.5 TD-MPC: Temporal Difference Learning for Model Predictive Control
- **Authors:** Hansen, Wang, Su
- **Year/Venue:** ICML 2022
- **Link:** https://arxiv.org/abs/2203.04955
- **Summary:** Learns an implicit (decoder-free) latent dynamics model jointly with a terminal value function via TD-learning. Uses MPPI for local trajectory optimization in latent space. Directly relevant to your MPC-based planning with a learned latent dynamics model.
- **Key Contribution:** Joint-embedding world model + TD-learned value function for MPC planning.

### 1.6 TD-MPC2: Scalable, Robust World Models for Continuous Control
- **Authors:** Hansen, Su et al.
- **Year/Venue:** ICLR 2025
- **Link:** https://arxiv.org/abs/2310.16828
- **Summary:** Scales TD-MPC to 104 tasks and 317M parameters with a single hyperparameter set. Demonstrates multi-task, multi-embodiment world model training. Relevant for scaling insights and the decoder-free latent dynamics + MPC paradigm.
- **Key Contribution:** Scalable multi-task world model; single agent for 80 tasks across embodiments.

### 1.7 FOWM: Finetuning Offline World Models in the Real World
- **Authors:** Feng, Hansen, Xiong, Rajagopalan, Wang
- **Year/Venue:** CoRL 2023
- **Link:** https://arxiv.org/abs/2310.16029
- **Summary:** Pretrains a world model on offline robot data, then finetunes with online planning. Balances estimated returns with epistemic model uncertainty during online rollouts. Relevant to your offline training pipeline and the filtered vs. unfiltered data comparison.
- **Key Contribution:** Offline-to-online world model finetuning with uncertainty-aware planning; few-shot adaptation on real robots.

### 1.8 UniSim: Learning Interactive Real-World Simulators
- **Authors:** Yang, Du, Ghasemipour, Tompson, Kaelbling, Schuurmans, Abbeel
- **Year/Venue:** ICLR 2024 (Outstanding Paper)
- **Link:** https://arxiv.org/abs/2310.06114
- **Summary:** Video diffusion world model trained on diverse data that simulates action-conditioned visual outcomes for both high-level instructions and low-level controls. Trained policies transfer zero-shot to the real world. Relevant as an exemplar of action-conditioned world models for robotics planning.
- **Key Contribution:** Universal action-conditioned video simulator; zero-shot sim-to-real transfer.

### 1.9 Robotic World Model: A Neural Network Simulator for Robust Policy Optimization
- **Authors:** Li, Krause, Hutter (ETH Zurich)
- **Year/Venue:** NeurIPS 2025
- **Link:** https://arxiv.org/abs/2501.10100
- **Summary:** Dual-autoregressive world model for partially observable, stochastic dynamics with self-supervised training. Policies trained in imagination transfer zero-shot to ANYmal D hardware. Directly relevant for long-horizon latent prediction and sim-to-real transfer.
- **Key Contribution:** Dual-autoregressive mechanism for robust long-horizon world model prediction; zero-shot real-robot deployment.

### 1.10 IRIS: Transformers are Sample-Efficient World Models
- **Authors:** Micheli et al.
- **Year/Venue:** ICLR 2023
- **Link:** https://arxiv.org/abs/2209.00588
- **Summary:** Uses a discrete autoencoder + autoregressive Transformer as a world model. Achieves superhuman Atari performance with only 2 hours of gameplay. Relevant as an alternative transformer-based dynamics architecture to the RSSM.
- **Key Contribution:** Transformer-based world model with discrete tokenized observations.

### 1.11 MILE: Model-Based Imitation Learning for Urban Driving
- **Authors:** Hu et al.
- **Year/Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2210.07729
- **Summary:** Jointly learns a compact latent world model and driving policy from offline expert video. Predicts future states in imagination and decodes to bird's-eye view. Relevant as a latent world model for vehicle state prediction with multi-step rollout from visual inputs.
- **Key Contribution:** Compact latent world model for autonomous driving with imagination-based planning.

### 1.12 GAIA-1: A Generative World Model for Autonomous Driving
- **Authors:** Hu, Russell, Yeo, Murez, Fedoseev, Kendall, Shotton, Corrado (Wayve)
- **Year/Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2309.17080
- **Summary:** 9B-parameter generative world model trained on 4,700 hours of driving data. Casts world modeling as next-token prediction conditioned on video, text, and action. Relevant for its action-conditioned latent prediction approach at scale.
- **Key Contribution:** Large-scale autoregressive world model for driving; action + language conditioned generation.

---

## 2. JEPA / Self-Supervised Prediction

### 2.1 I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- **Authors:** Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas
- **Year/Venue:** CVPR 2023
- **Link:** https://arxiv.org/abs/2301.08243
- **Summary:** Predicts representations of masked image patches in embedding space (not pixel space). Learns semantic features without data augmentation or pixel reconstruction. Foundational to the JEPA paradigm your project draws from for latent prediction rather than pixel-level reconstruction.
- **Key Contribution:** Non-generative self-supervised learning via abstract representation prediction.

### 2.2 V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video
- **Authors:** Bardes et al. (Meta FAIR)
- **Year/Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2404.08471
- **Summary:** Extends JEPA to video with spatiotemporal masking and feature prediction. Learns versatile visual representations for both motion and appearance tasks without any labels, text, or reconstruction. Relevant as the video-level self-supervised prediction paradigm your latent dynamics model instantiates.
- **Key Contribution:** Self-supervised video representation learning via latent feature prediction.

### 2.3 V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **Authors:** Bardes et al. (Meta FAIR)
- **Year/Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2506.09985
- **Summary:** Scales V-JEPA to 1M+ hours of video. Introduces an action-conditioned world model (300M-parameter transformer) for robotic planning on Franka arms with zero-shot transfer. Directly relevant as a state-of-the-art JEPA world model applied to robot manipulation planning.
- **Key Contribution:** Action-conditioned JEPA world model for zero-shot robotic planning.

### 2.4 MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features
- **Authors:** Bardes, Ponce et al.
- **Year/Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2307.12698
- **Summary:** Jointly learns optical flow and content features within a shared encoder using JEPA-style prediction. Shows that motion estimation and self-supervised learning objectives mutually benefit each other. Relevant for learning motion-aware representations from video sequences.
- **Key Contribution:** Joint motion + content feature learning in a shared JEPA encoder.

---

## 3. Vision Backbones for Robotics

### 3.1 DINOv2: Learning Robust Visual Features without Supervision
- **Authors:** Oquab et al. (Meta FAIR)
- **Year/Venue:** TMLR 2024
- **Link:** https://arxiv.org/abs/2304.07193
- **Summary:** Self-supervised ViT trained on curated diverse data that produces general-purpose visual features competitive with supervised models across tasks. Used as a frozen visual backbone in your project. Relevant as one of the two visual encoder options (DINOv2 vs CNN).
- **Key Contribution:** Universal visual features from self-supervised learning on curated large-scale data.

### 3.2 VC-1: Where Are We in the Search for an Artificial Visual Cortex for Embodied Intelligence?
- **Authors:** Majumdar et al. (Meta FAIR)
- **Year/Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2303.18240
- **Summary:** Largest empirical study of pre-trained visual representations for Embodied AI across 17 tasks (CortexBench). Trains VC-1 (ViT + MAE) on 4,000+ hours of egocentric video. Relevant for understanding which visual backbones transfer best to embodied robot tasks.
- **Key Contribution:** CortexBench evaluation suite; VC-1 pretrained visual model for embodied AI.

### 3.3 R3M: A Universal Visual Representation for Robot Manipulation
- **Authors:** Nair, Rajeswaran et al.
- **Year/Venue:** CoRL 2022
- **Link:** https://arxiv.org/abs/2203.12601
- **Summary:** Pre-trains visual representations on diverse human video using time-contrastive learning + video-language alignment. Enables data-efficient robot manipulation with only 20 demonstrations. Relevant as a competing visual backbone approach to DINOv2 for robot learning.
- **Key Contribution:** Video-pretrained visual representations enabling few-shot robot manipulation.

### 3.4 MVP: Masked Visual Pre-training for Motor Control
- **Authors:** Xiao, Radosavovic, Darrell, Malik
- **Year/Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2203.06173
- **Summary:** Shows masked autoencoder pre-training on real-world images yields visual representations that outperform supervised encoders by up to 80% for motor control. Frozen encoder + RL policy. Directly relevant to the frozen visual backbone + downstream dynamics architecture of race16.
- **Key Contribution:** Self-supervised MAE pretraining for robot motor control; frozen encoder paradigm.

---

## 4. Aerial/Ground Robot Learning

### 4.1 Champion-Level Drone Racing Using Deep Reinforcement Learning (Swift)
- **Authors:** Kaufmann, Bauersfeld, Loquercio et al.
- **Year/Venue:** Nature 2023
- **Link:** https://www.nature.com/articles/s41586-023-06419-4
- **Summary:** Autonomous drone system that beats world-champion pilots using deep RL trained in simulation with real-world data. Combines learned perception and control for agile flight. Relevant as the state-of-the-art in learned autonomous drone control with sim-to-real transfer.
- **Key Contribution:** First autonomous system to beat human champions in physical drone racing.

### 4.2 Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight
- **Authors:** Romero, Shenai et al. (UZH RPG)
- **Year/Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2501.14377
- **Summary:** Applies DreamerV3 to autonomous drone racing, learning to fly directly from raw pixel observations without intermediate state estimation. First approach to learn drone racing from pixels end-to-end. Highly relevant as a world model applied to aerial vehicle control from vision.
- **Key Contribution:** DreamerV3-based world model for vision-only drone racing; no explicit state estimation needed.

### 4.3 Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms
- **Authors:** Salzmann, Kaufmann, Arrizabalaga, Pavone, Scaramuzza, Ryll
- **Year/Venue:** IEEE RA-L 2023
- **Link:** https://arxiv.org/abs/2203.07747
- **Summary:** Integrates large neural network dynamics models into real-time MPC for quadrotors. Achieves 50Hz control on embedded hardware with 82% tracking error reduction over classical MPC. Directly relevant as learned dynamics + MPC on a physical aerial platform.
- **Key Contribution:** Efficient neural dynamics integration in real-time MPC for agile quadrotor control.

### 4.4 SafeDreamer: Safe Reinforcement Learning with World Models
- **Authors:** Huang, Ji et al.
- **Year/Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2307.07176
- **Summary:** Integrates Lagrangian-based safety constraints into Dreamer world model planning. Achieves near-zero constraint violations from both state and vision inputs. Relevant to the risk/collision prediction head in race16 -- SafeDreamer shows how world models can incorporate safety constraints.
- **Key Contribution:** Safety-constrained world model planning with Lagrangian methods.

### 4.5 TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers
- **Authors:** Nguyen et al.
- **Year/Venue:** ICRA 2024 (Best Paper in Automation)
- **Link:** https://arxiv.org/abs/2310.16985
- **Summary:** ADMM-based MPC solver achieving 8x speedup over OSQP on a 27g quadrotor microcontroller. Demonstrates trajectory tracking and obstacle avoidance at high rates on embedded hardware. Relevant to deploying MPC-based planning on the race16 hardware platform.
- **Key Contribution:** Ultra-efficient MPC for resource-constrained robotic platforms.

---

## 5. MPC with Learned Models

### 5.1 TD-MPC / TD-MPC2 (listed above in Section 1.5 and 1.6)

### 5.2 COMBO: Conservative Offline Model-Based Policy Optimization
- **Authors:** Yu, Kumar, Rafailov, Rajeswaran, Levine, Finn
- **Year/Venue:** NeurIPS 2021
- **Link:** https://arxiv.org/abs/2102.08363
- **Summary:** Offline model-based RL that trains a value function on both real and model-generated data with conservative regularization, without requiring explicit uncertainty quantification. Relevant for offline training of world models from logged robot data with conservative estimates.
- **Key Contribution:** Conservative value estimation for offline model-based RL without uncertainty quantification.

---

## 6. Data Curation for Robot Learning

### 6.1 Data Quality in Imitation Learning
- **Authors:** Mandlekar et al.
- **Year/Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2306.02437
- **Summary:** Formalizes data quality for imitation learning through the lens of distribution shift. Identifies two key dataset properties: action divergence and transition diversity. Shows that state diversity is not always beneficial. Directly relevant to your filtered vs. unfiltered training data comparison.
- **Key Contribution:** Formal framework for understanding data quality in imitation learning; actionable data filtering criteria.

### 6.2 Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **Authors:** Open X-Embodiment Collaboration
- **Year/Venue:** ICRA 2024
- **Link:** https://arxiv.org/abs/2310.08864
- **Summary:** 1M+ real robot trajectories across 22 embodiments from 34 labs. Shows that cross-embodiment training improves generalization (RT-1-X 50% better, RT-2-X 3x better on new tasks). Relevant for understanding how dataset scale and diversity impact robot learning performance.
- **Key Contribution:** Largest cross-embodiment robot dataset; evidence that data diversity improves generalization.

### 6.3 Robot Data Curation with Mutual Information Estimators
- **Authors:** Hejna et al.
- **Year/Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2502.08623
- **Summary:** Uses k-nearest-neighbor mutual information estimation on VAE embeddings to measure each episode's contribution to dataset quality. Achieves 5-10% improvement on RoboMimic and real robot setups. Relevant to developing principled data filtering strategies for your training pipeline.
- **Key Contribution:** Mutual information-based episode scoring for robot demonstration curation.

### 6.4 DataMIL: Selecting Data for Robot Imitation Learning with Datamodels
- **Authors:** UT Austin RobIn Lab
- **Year/Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2505.09603
- **Summary:** Policy-driven data selection framework that uses the policy itself to identify which data points improve performance, rather than human-defined quality heuristics. 10% performance boost on MetaWorld tasks. Relevant to principled approaches for selecting training data subsets.
- **Key Contribution:** End-to-end data selection optimized for downstream task success rather than proxy metrics.

---

## Summary Table

| # | Paper | Year | Venue | Category |
|---|-------|------|-------|----------|
| 1 | PlaNet (Hafner et al.) | 2019 | ICML | World Models |
| 2 | DreamerV2 (Hafner et al.) | 2021 | ICLR | World Models |
| 3 | DreamerV3 (Hafner et al.) | 2023 | ICLR / Nature | World Models |
| 4 | DayDreamer (Wu et al.) | 2022 | CoRL | World Models |
| 5 | TD-MPC (Hansen et al.) | 2022 | ICML | World Models / MPC |
| 6 | TD-MPC2 (Hansen et al.) | 2025 | ICLR | World Models / MPC |
| 7 | FOWM (Feng et al.) | 2023 | CoRL | World Models |
| 8 | UniSim (Yang et al.) | 2024 | ICLR | World Models |
| 9 | Robotic World Model (Li et al.) | 2025 | NeurIPS | World Models |
| 10 | IRIS (Micheli et al.) | 2023 | ICLR | World Models |
| 11 | MILE (Hu et al.) | 2022 | NeurIPS | World Models / Driving |
| 12 | GAIA-1 (Hu et al.) | 2023 | arXiv | World Models / Driving |
| 13 | I-JEPA (Assran et al.) | 2023 | CVPR | JEPA |
| 14 | V-JEPA (Bardes et al.) | 2024 | ICML | JEPA |
| 15 | V-JEPA 2 (Bardes et al.) | 2025 | arXiv | JEPA |
| 16 | MC-JEPA (Bardes et al.) | 2023 | NeurIPS | JEPA |
| 17 | DINOv2 (Oquab et al.) | 2024 | TMLR | Vision Backbone |
| 18 | VC-1 (Majumdar et al.) | 2023 | NeurIPS | Vision Backbone |
| 19 | R3M (Nair et al.) | 2022 | CoRL | Vision Backbone |
| 20 | MVP (Xiao et al.) | 2022 | NeurIPS | Vision Backbone |
| 21 | Swift (Kaufmann et al.) | 2023 | Nature | Aerial Robot |
| 22 | Dream to Fly (Romero et al.) | 2025 | arXiv | Aerial Robot |
| 23 | Neural-MPC (Salzmann et al.) | 2023 | RA-L | Aerial / MPC |
| 24 | SafeDreamer (Huang et al.) | 2024 | ICLR | World Models / Safety |
| 25 | TinyMPC (Nguyen et al.) | 2024 | ICRA | MPC / Embedded |
| 26 | COMBO (Yu et al.) | 2021 | NeurIPS | MPC / Offline RL |
| 27 | Data Quality in IL (Mandlekar et al.) | 2023 | NeurIPS | Data Curation |
| 28 | Open X-Embodiment | 2024 | ICRA | Data / Cross-Embodiment |
| 29 | Robot Data Curation (Hejna et al.) | 2025 | arXiv | Data Curation |
| 30 | DataMIL (UT Austin) | 2025 | arXiv | Data Curation |

---

## Notes on Link Reliability

All arXiv links above use confirmed paper IDs from search results. Venue/year information was cross-checked against Semantic Scholar, OpenReview, and official proceedings pages. For papers listed as "arXiv" venue, this indicates the paper was found as a preprint without confirmed peer-review venue at the time of compilation.

# WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking

Code & dataset repository for the paper: **[Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://arxiv.org/abs/2311.05237)**

Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa

[![arXiv](https://img.shields.io/badge/arXiv-2311.05237-00ff00.svg)](https://arxiv.org/abs/2311.05237) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

We present Widely Applicable Strong Baseline (WASB), a Sports Ball Detection and Tracking (SBDT) baseline that can be applied to wide range of sports categories :soccer: :tennis: :badminton: :volleyball: :basketball: .

https://github.com/nttcom/WASB-SBDT/assets/63090948/8889ef53-62c7-4c97-9b33-8bf386489ba1

## News

- [11/23/2023] [Our BMVC2023 proceeding](https://proceedings.bmvc2023.org/310/) is available! Thank you, BMVC2023 organizers!
- [11/23/2023] Evaluation codes of DeepBall, DeepBall-Large and BallSeg are added!
- [11/21/2023] Evaluation codes of TrackNetV2, ResTrackNetV2 and MonoTrack are added!
- [11/17/2023] Repository is released. Now it contains evaluation codes of pretrained WASB models only. Other models will be coming soon!
- [11/09/2023] Our [arXiv preprint](https://arxiv.org/abs/2311.05237) is released.

## Installation and Setup

Tested with Python3.8, CUDA11.3 on Ubuntu 18.04 (4 V100 GPUs inside). We recommend to use the [Dockerfile](./Dockerfile) provided in this repo (with ```-it``` option when running the container). 

- See [GET_STARTED.md](./GET_STARTED.md) for how to get started with SBDT models.
- See [MODEL_ZOO.md](./MODEL_ZOO.md) for available model weights.

## Citation

If you find this work useful, please consider to cite our paper:

```
@inproceedings{tarashima2023wasb,
	title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
	author={Tarashima, Shuhei and Haq, Muhammad Abdul and Wang, Yushan and Tagawa, Norio},
	booktitle={BMVC},
	year={2023}
}
```


## Extending WASB-SBDT with a New Model (Example: HRCNet)

This repository is designed so that new backbone models can be plugged into the training and evaluation pipeline with minimal changes. Below is a high-level checklist based on the integration of **HRCNet**.

1. **Implement the model and a small wrapper (if necessary)**
   - Add your model implementation under `src/models/` (e.g. `hrcnet.py`).
   - If the framework expects a specific output format (e.g. a dict of heatmaps per scale), provide a thin wrapper class (e.g. `HRCNetForWASB`) whose `forward` matches the existing models (e.g. `return {0: heatmap}`), while the base model can still return richer diagnostic outputs.

2. **Register the model in the model factory**
   - Import the model (or wrapper) in `src/models/__init__.py`.
   - Add an entry to `__factory`, e.g. `"hrcnet": HRCNetForWASB`.
   - Extend `build_model(cfg)` with a new branch that:
     - Reads all required hyperparameters from `cfg['model']`.
     - Constructs the model instance with proper input/output channel sizes (e.g. `in_channels = frames_in * 3`, `out_channels = frames_out`).

3. **Add Hydra configs for the new model and training**
   - Create `src/configs/model/<your_model>.yaml` (e.g. `hrcnet.yaml`):
     - Set `name: <your_model_name>` and all model hyperparameters (`frames_in`, `frames_out`, spatial resolutions, internal channels, transformer/dowmsampling settings, etc.).
   - Ensure `src/configs/train.yaml` includes the desired defaults (dataset, model, loss, optimizer, detector, transform, tracker) and that `_self_` appears at the end of the `defaults` list so overrides in `train.yaml` (e.g. `dataloader.train: True`) are applied last.
   - Use `src/configs/runner/train.yaml` to control training behaviour (device, epochs, test/inference flags, checkpoint naming).

4. **Make the dataloader aware of the new model**
   - In `src/dataloaders/__init__.py`, `build_dataloader` selects how to build datasets based on `cfg['model']['name']`.
   - Add your model name to the allowed list so that it follows the same `ImageDataset` pipeline as the existing heatmap-based models, e.g.:
     - `if model_name in ['tracknetv2', 'hrnet', 'hrcnet', 'monotrack', 'restracknetv2', 'deepball', 'ballseg']:`

5. **Update detectors / postprocessors if needed**
   - In `src/detectors/detector.py`, `TracknetV2Detector` validates `cfg['model']['name']` and expects a specific prediction format.
   - Add the new model name into this validation list and ensure your model outputs are compatible with the existing postprocessor (e.g. the same dict-of-heatmaps interface).

6. **Hook into runners (training / evaluation)**
   - Make sure `src/runners/__init__.py` registers the training and evaluation runners:
     - `"train": Trainer`, `"eval": VideosInferenceRunner`, etc.
   - `Trainer` uses `build_model(cfg)` and `build_dataloader(cfg)`, so once both are updated, it can train the new model without changes.
   - `VideosInferenceRunner` should accept an optional `model` argument so that training-time video inference can reuse the in-memory model instead of reloading from disk.

7. **Verify training & evaluation commands**
   - Run a short training job to verify the pipeline end-to-end (e.g. a few epochs):

     ```bash
     cd src
     python main.py \
       --config-name=train \
       dataset=tennis \
       model=hrcnet \
       loss=hm_bce \
       optimizer=adam_multistep \
       detector=tracknetv2 \
       transform=default \
       tracker=online \
       output_dir=../outputs/hrnet_vs_hrcnet/hrcnet
     ```

   - Evaluate the trained checkpoint with `--config-name=eval` and compare the resulting metrics (Precision, Recall, F1, Accuracy, RMSE, AP) against existing models.



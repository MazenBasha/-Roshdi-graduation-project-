# Face Recognition Model Training Report

## Dataset
- **Training/Validation Directory:** lfw_funneled
- **Number of Classes:** 5749

## Training Configuration
- **Epochs:** 50
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Embedding Size:** 128
- **Input Size:** 112
- **ArcFace Margin:** 0.5
- **ArcFace Scale:** 64
- **Training Date:** 2026-03-07

## Training Results
### Final Metrics
- **Final Training Loss:** 7.1455
- **Final Training Accuracy:** 10.13%
- **Final Validation Loss:** 7.4781
- **Final Validation Accuracy:** 6.09%
- **Best Validation Accuracy:** 7.89%

### Training History (first and last 5 epochs)
#### Loss
- Start: 8.5821, 8.1570, 8.0419, 7.9761, 7.9162
- End: 7.1508, 7.1455

#### Accuracy
- Start: 3.61%, 4.00%, 4.01%, 4.04%, 4.10%
- End: 10.18%, 10.13%

#### Validation Loss
- Start: 8.2207, 8.0279, 7.9598, 7.9029, 7.8528
- End: 7.4086, 7.4781

#### Validation Accuracy
- Start: 4.01%, 4.01%, 4.01%, 4.04%, 4.08%
- End: 7.25%, 6.09%

### Full Training/Validation Curves
See `checkpoints/training_history.json` for all epoch-by-epoch values.

## Model Artifacts
- **Best Model Weights:** checkpoints/best_model.h5
- **Backbone Model:** checkpoints/backbone.h5
- **Full Model:** checkpoints/training_model.h5
- **Training History:** checkpoints/training_history.json
- **Training Config:** checkpoints/config.json

## Notes
- Training and validation were both performed on the lfw_funneled dataset.
- For further analysis, refer to the JSON files for complete metrics.
- For inference or evaluation, use the saved backbone or full model.

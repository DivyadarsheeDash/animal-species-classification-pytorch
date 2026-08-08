# Animal Species Classification with PyTorch

A **transfer-learning practice project** built from Animal-CLEF 2026-related image data. The goal of this repository is to strengthen practical PyTorch skills for multiclass image classification rather than claim an official competition result.

## Project Motivation

This project was created specifically to practice:

- PyTorch image pipelines
- Data augmentation
- Transfer learning
- EfficientNet architectures
- Training/evaluation loops
- Saving trained model weights

The working dataset contains image folders associated with:

- `LynxID2025`
- `SalamanderID2025`
- `SeaTurtleID2022`
- `TexasHornedLizards`

A custom held-out split was created for experimentation; in particular, the Texas horned lizard data required manual train/test organization.

## Important Benchmark Note

The reported results are from a **custom practice split** and are **not directly comparable to the official Animal-CLEF competition benchmark**.

## Modeling

The notebook experiments with EfficientNet variants using transfer learning.

Recorded experiments include:

- EfficientNet-B0
- EfficientNet-B3
- EfficientNet-B4

The pipeline replaces the final classifier to match the project classes and fine-tunes the network using PyTorch.

## Recorded Results

On the custom held-out split, the strongest recorded run in the notebook is from **EfficientNet-B3**, reaching approximately:

**99.96% test accuracy**

at the best recorded epoch.

Other architectures/runs show different behavior, including a lower result from the larger B4 experiment, which is a useful reminder that a larger model is not automatically better.

## Repository Structure

The main public notebook contains:

- Dataset loading
- Train/test preparation
- DataLoaders
- Transfer-learning models
- Training and testing loops
- Architecture comparisons
- Model serialization

The trained state dict is saved in the notebook workflow as:

```text
animal_classifier.pth
```

## Tech Stack

- Python
- PyTorch
- TorchVision
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib

## Lessons from the Project

- Transfer learning can outperform small CNNs trained from scratch when the dataset is limited.
- Model size should be validated empirically.
- A high score on a custom split does not establish competition-level generalization.
- Dataset splitting is as important as architecture choice.

## Limitations

- Custom split rather than official challenge evaluation.
- No claim of competition leaderboard performance.
- More rigorous identity/source-aware splitting may be required depending on the original dataset structure.
- Multi-seed experiments are not currently reported.

## Future Work

- Rebuild the split with a fully documented protocol.
- Report macro F1 and class-level recall.
- Add confusion matrix.
- Add reproducible random seeds.
- Compare frozen-backbone vs full fine-tuning.
- Evaluate augmentations systematically.
- Add official benchmark evaluation if competition labels/protocol permit.

## Author

**Divyadarshee Dash**

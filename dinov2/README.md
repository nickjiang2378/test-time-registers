# DINOv2

This folder was modified from the official DINOv2 [repo](https://github.com/facebookresearch/dinov2). Here are the primary file modifications we make to add support for test-time registers. All modifications are marked. (search for "Start of custom code")
1. `dinov2/models/vision_transformer.py`: we add a section in `DinoVisionTransformer.prepare_tokens_with_masks` to create test-time registers initialized to the mean of the input image patches.
2. `dinov2/layers/mlp.py`: we add an identity function that we use in our hooks to extract MLP neuron activations for our analysis.
3. `dinov2/layers/swiglu_ffn.py`: we add an identity function to extract MLP neuron activations for our analysis. Swiglu is used for larger backbone sizes.
4. `dinov2/layers/attention.py`, `dinov2/layers/block.py`, `dinov2/layers/swiglu_ffn.py`: disable xformers for easier analysis

## Model environment

To create an environment just for DINOv2, create a new conda environment and install the packages in `requirements.txt`. Then, run the analysis for DINOv2 in `register_neurons.ipynb`.
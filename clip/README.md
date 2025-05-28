# CLIP

This folder was modified from this [repo](https://github.com/yossigandelsman/second_order_lens), which provides access to both [OpenCLIP](https://github.com/mlfoundations/open_clip) and [OpenAI](https://github.com/openai/CLIP) models. Here are the file modifications we make to add support for test-time registers. All modifications are marked.
1. Rename the `utils` folder to `clip`.
2. `clip/transformer.py`: we add extra tokens to serve as "test-time" registers during the forward pass.

## Model environment

To create an environment just for CLIP, execute `conda env create -f environment.yml`. Then, run the analysis for CLIP in `register_neurons.ipynb`.
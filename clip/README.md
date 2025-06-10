# CLIP

This folder was modified from this [repo](https://github.com/yossigandelsman/second_order_lens), which provides access to both [OpenCLIP](https://github.com/mlfoundations/open_clip) and [OpenAI](https://github.com/openai/CLIP) models. Here are the primary file modifications we make to add support for test-time registers. All modifications are marked (search for "Start of custom code")
1. Rename the `utils` folder to `clip`.
2. `clip/transformer.py`: we add extra tokens to serve as "test-time" registers during the forward pass.


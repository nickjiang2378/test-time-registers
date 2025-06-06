# Vision Transformers Don't Need Trained Registers

[Nick Jiang](https://nickjiang.me)\*, [Amil Dravid](https://avdravid.github.io/)\*, [Alexei Efros](https://people.eecs.berkeley.edu/~efros/), [Yossi Gandelsman](https://yossigandelsman.github.io/)

[Paper]() | [Project Page]()

![Teaser Figure](plots/Teaser.png)

**Controlling high-norm tokens in Vision Transformers.** As shown in [Darcet et al. (2024)](https://arxiv.org/abs/2309.16588), high-norm outlier tokens emerge in ViTs and lead to noisy attention maps (“Original”). By identifying the mechanism responsible for their emergence, we demonstrate that we can shift them to arbitrary positions at test time (“Shifted”). Shifting the outlier tokens outside of the image area mimics register behavior at test-time (“w/ Test-time Register”), resulting in more interpretable attention patterns and downstream performance comparable to models retrained with registers.

## Setup

### Repo
```
git clone git@github.com:nickjiang2378/test-time-registers.git
cd test-time-registers
```

### Environment

```
conda env create -f environment.yml
```

## Repo Overview
```
test-time-registers/
├── clip/
│   ├── ...
│   ├── clip_hook_manager.py
│   └── clip_state.py
├── dinov2/
│   ├── ...
│   ├── dinov2_hook_manager.py
│   └── dinov2_state.py
└── shared/
    ├── ...
    ├── algorithms.py
    ├── hook_manager.py
    └── hook_fn.py
```
Here are the most important files in this repo:

`hook_manager.py`: manages all hooks (interventions, logging) registered for the model. `CLIPHookManager` and `Dinov2HookManager` are both subclasses.

`hook_fn.py`: contains the hook functions for intervening on register neurons and logging model internals for analysis

`algorithms.py`: contains algorithm for detecting register neurons

`clip_state.py` / `dinov2_state.py`: loads model, instantiates hook manager, and passes important metadata like number of layers

## Register Neuron Discovery

See `register_neurons.ipynb` to automatically find register neurons and analyze the effects of intervening upon them with test-time registers.

### ImageNet

Many sections rely on access to a corpus of images for use in analysis; our paper uses [ImageNet 2012](https://www.image-net.org/challenges/LSVRC/index.php). You can also create a folder of custom images, but the folder should only consist of images (ie. JPEGs). Once collected or downloaded, pass the folder path to the `IMAGENET_PATH` variable in the notebook.

### Adding New Models

To study ViTs beyond CLIP and DINOv2, we recommend creating a new environment to avoid dependency conflicts with DINOv2 and CLIP. Our method primarily requires a modern version of Pytorch (see `shared/requirements.txt`). Copy the new model's code into this repo (similar to what's done with CLIP and DINOv2), and create two additional files in the new model's folder:
1. `custom_hook_manager.py`: initialize a subclass of the `HookManager` class in `shared/hook_manager.py` and fill out the abstract methods, which tell us how to hook into important model components like the MLP, attention heads, etc. See `dinov2/dinov2_hook_manager.py` for an example.
2. `custom_state.py`: create a `load_model_state` function that loads in a model based on a config (to specify size, etc.), instantiates the hook manager, and returns metadata like number of layers. See `dinov2/dinov2_state.py` for an example.

Lastly, you should modify the model code to enable adding in extra tokens initialized to zeros. This is necessary for creating our "test-time" registers to shift outliers from the image to. In CLIP, we pass in the number of registers during the forward pass. In DINOv2, we set this number as an attribute of the model class. See their respective folders for more details.


# Ready-to-Use Models with Test-Time Registers

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />linear</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="right">21 M</td>
      <td align="center">:x:</td>
      <td align="right">79.0%</td>
      <td align="right">81.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="right">21 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">79.1%</td>
      <td align="right">80.9%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="right">86 M</td>
      <td align="center">:x:</td>
      <td align="right">82.1%</td>
      <td align="right">84.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="right">86 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">82.0%</td>
      <td align="right">84.6%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.3%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.8%</td>
      <td align="right">86.7%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.7%</td>
      <td align="right">87.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
  </tbody>
</table>

import os
from pathlib import Path

# Get the directory where hubconf.py is located and change to dinov2 subdirectory
hub_dir = Path(__file__).parent.absolute()
import sys
sys.path.append(os.path.join(hub_dir, "dinov2"))

from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from dinov2.hub.backbones import dinov2_vitl14_tt_reg
from dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg
from dinov2.hub.classifiers import dinov2_vitb14_lc, dinov2_vitg14_lc, dinov2_vitl14_lc, dinov2_vits14_lc
from dinov2.hub.classifiers import dinov2_vitb14_reg_lc, dinov2_vitg14_reg_lc, dinov2_vitl14_reg_lc, dinov2_vits14_reg_lc
from dinov2.hub.depthers import dinov2_vitb14_ld, dinov2_vitg14_ld, dinov2_vitl14_ld, dinov2_vits14_ld
from dinov2.hub.depthers import dinov2_vitb14_dd, dinov2_vitg14_dd, dinov2_vitl14_dd, dinov2_vits14_dd

dependencies = ["torch"]

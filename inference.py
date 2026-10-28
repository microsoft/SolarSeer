import numpy as np
import torch
from network.SolarSeerNet import SolarSeerNet
import time
import os
from pathlib import Path

# network and hyperparameters
start_time = time.perf_counter()
base_dir = Path(__file__).resolve().parent
multi_params = [{
"img_size": [512, 1280],
"embed_dim": 600,
"depth": 4,
"mlp_ratio": 4.0,
"drop_rate": 0.01,
"drop_path_rate": 0.01,
"num_blocks": 6,
"sparsity_threshold": 0.01,
"hard_thresholding_fraction": 1.0,
"input_time_dim": 6,
"output_time_dim": 24,
"autoregressive_steps": 1,
"use_dilated_conv_blocks": True,
"output_only_last": False,
"patch_size": 4,
"N_in_channels": 4,
"N_out_channels": 1,
"target_size": [512, 1280],
"target_variable_index": None,
"topo_index": None,
"new_index": None
}]
model = SolarSeerNet(multi_params=multi_params,
                        act_final="Tanh",
                        use_dilated_conv_blocks=False,
                        autoregressive_steps=1,
                        target_variable_index=[0],
                        action="add")
state_dict = torch.load(
    base_dir / 'weight' / 'SolarSeer.pt',
    map_location='cpu',
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# input data
# sample data. The issue time is UTC 2023-10-11 06:00:00
# the sample data is available at Baidu Disk.
date = 'UTC 2023-10-11 06:00:00'
satellite = torch.from_numpy(np.load(base_dir / 'input' / 'satellite.npy')).float().to(device)
clearghi = torch.from_numpy(np.load(base_dir / 'input' / 'clearghi.npy')).float().to(device)

# output
with torch.inference_mode():
    y = model(satellite, clearghi).cpu().numpy()
y[:, 0, :, :] = y[:, 0, :, :] * 1000 # surface solar irradiance, W/m^2
y[:, 1, :, :] = y[:, 1, :, :] * 100 # total cloud cover, 0%~100%

# time  
end_time = time.perf_counter()
print(end_time - start_time)

# save results
results_dir = base_dir / 'results'
os.makedirs(results_dir, exist_ok=True)
np.save(results_dir / 'irradiance_cloud.npy', y)
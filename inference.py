import numpy as np
import torch
from network.SolarSeerNet import SolarSeerNet
import time
import os

# network and hyperparameters
start_time = time.perf_counter()
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
model = torch.load('./weight/SolarSeer.pt', weights_only=False)
model.eval()

# input data
# random data
satellite = torch.rand(1, 4, 6, 512, 1280).type(torch.float32)
clearghi = torch.rand(24, 480, 1150).type(torch.float32)
# sample data. The issue time is UTC 2023-10-11 06:00:00
# the sample data is available at Baidu Disk. 
date = 'UTC 2023-10-11 06:00:00'
satellite = torch.from_numpy(np.load('input/satellite.npy')).float()
clearghi = torch.from_numpy(np.load('input/clearghi.npy')).float()  
satellite.to('cuda:0')
clearghi.to('cuda:0')

# output
y = model(satellite, clearghi).cpu().detach().numpy()
y[:, 0, :, :] = y[:, 0, :, :] * 1000 # surface solar irradiance, W/m^2
y[:, 1, :, :] = y[:, 1, :, :] * 100 # total cloud cover, 0%~100%

# time  
end_time = time.perf_counter()
print(end_time - start_time)

# save results
os.makedirs('results', exist_ok=True)
np.save('results/irradiance_cloud.npy', y)   
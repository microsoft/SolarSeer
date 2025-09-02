SolarSeer
---
This repository contains the code used for the [paper] https://arxiv.org/abs/2508.03590 entitled "SolarSeer: Ultrafast and accurate 24-hour solar irradiance forecasts
outperforming numerical weather prediction across the USA" 

Abstract
---
Accurate 24-hour solar irradiance forecasting is essential for the safe and economic operation of solar
photovoltaic systems. Traditional numerical weather prediction (NWP) models represent the state-of-the-art in
forecasting performance but rely on computationally costly data assimilation and solving complicated partial
differential equations (PDEs) that simulate atmospheric physics. Here, we introduce SolarSeer, an end-to-end large
artificial intelligence (AI) model for solar irradiance forecasting across the Contiguous United States (CONUS).
SolarSeer is designed to directly map the historical satellite observations to future forecasts, eliminating the
computational overhead of data assimilation and PDEs solving. This efficiency allows SolarSeer to operate over
1,500 times faster than traditional NWP, generating 24-hour cloud cover and solar irradiance forecasts for the
CONUS at 5-kilometer resolution in under 3 seconds. Compared with the state-of-the-art NWP in the CONUS,
i.e., High-Resolution Rapid Refresh (HRRR), SolarSeer significantly reduces the root mean squared error of solar
irradiance forecasting by 27.28% in reanalysis data and 15.35% across 1,800 stations. SolarSeer also effectively
captures solar irradiance fluctuations and significantly enhances the first-order irradiance difference forecasting
accuracy. SolarSeer’s ultrafast, accurate 24-hour solar irradiance forecasts provide strong support for the transition
to sustainable, net-zero energy systems.

Network
---
SolarSeer is an end-to-end AI model trained with 16 MI200 GPUs for 24-hour-ahead solar irradiance forecasting across the USA. The network input is the past 6-hour 
satellite observations and the future 24-hour clear-sky solar irradiance, where clear-sky solar irradiance is a function of latitude, 
longtitude and time. The network output is the future 24-hour-ahead solar irradiance forecasts. SolarSeer can serve as a foundation 
model for solar PV power generation forecasting. The network structure of SolarSeer is as follows:

<img width="514" height="567" alt="image" src="https://github.com/user-attachments/assets/a8621cc2-e80b-4b3a-bbb2-a3eb03ca55c6" />

For the network of SolarSeer, see 'network/SolarSeerNet.py'.

Inference
---
To run the inference code, please follow the steps below:

1. Download the pretrained network weights and put it in the "weight" folder.
2. Download the network input, namely "satellite.npy" and "clearghi.npy". Put the two files in the "input" folder.
   The input file is available at [BaiduDisk] (https://pan.baidu.com/s/1i91AQ01BJhuW1BYkpES19Q).
   The pretrained weight file is available at [BaiduDisk] (https://pan.baidu.com/s/1QzEO_in6k9IekGcfp8qTCA).
   For the password of these files, please contact mingliangbai@outlook.com. 
4. Install packages using the command ```pip install -r requirement.txt ``` .
5. Run the 'inference.py' file. The generated forecasts will be saved in 'results' folder.

Reference
---
If you find it useful, cite it using:

```bibtex
@article{bai2025solarseer,
  title={SolarSeer: Ultrafast and accurate 24-hour solar irradiance forecasts 
                    outperforming numerical weather prediction across the USA},
  author={Bai, Mingliang and Fang, Zuliang and Tao, Shengyu and Xiang, Siqi and Bian, 
          Jiang and Xiang, Yanfei and Zhao, Pengcheng and Jin, Weixin 
          and Weyn, Jonathan A and Dong, Haiyu and others},
  journal={arXiv preprint arXiv:2508.03590},
  year={2025}
}

# OSL Simulator
Wrapper for launching simulated missions. The repository is used for launching the Gazebo simulation using UUV simulator to run OSL worlds for collecting underwater data and for runnig Yolo preidiction to auto detection the objects.

### Data Collection
For running the simulator for collecting data run following in one terminal,
```sh
roslaunch osl_simulator osl_wavetank.launch world:=worlds/dataset.world
```
In another terminal,
```sh
roslaunch osl_simulator send_waypoints.launch
```
Once the waypoints have been sent, use the same termimal to set the data collection rosparam to true. 
```sh
rosparam set /gt_save true
```
The data is stored in 'osl_network/data' folder. To add variablity to the data, modify 'world/dataset.world' or 'world/wavetank.world' and move around the models in different locations. You can also modify 'config/waypoints_*' files to change the path of the vehicle during the colleciton run. 

### For Prediction
Use https://www.dropbox.com/sh/37rfq3qcrsmrf25/AAANtod91ME3et4XF6FmhqzJa?dl=0 for downloading models and add them to osl_network in the workspace. DO NOT UPLOAD MODELS ON GIT for file size issues. For prediction, run the simulator with prediction flag on. 
```sh
roslaunch osl_simulator osl_wavetank.launch world:=worlds/wavetank.world yolo_predict:=true
```

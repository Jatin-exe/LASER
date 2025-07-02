# Documentation and clean up coming soon 

## Very minimal

To start run this

```shell
bash start_container_offline.sh
```

You may need to change paths, it is a good starting point though.

Visit [NVIDIA ISAAC ROS](https://nvidia-isaac-ros.github.io/index.html) for full setup.

Enter workspace and compile

```shell
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)
```

You can start playing around more now.

We will periodically update the documentation and add demos.

Feel free to open any issues, and we will work through them one by one.
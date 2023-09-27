```
mkdir TRT
cd TRT
git clone -b master https://github.com/NVIDIA/TensorRT TensorRT
cd TensorRT/
ls -la
git checkout release/8.2
git submodule update --init --recursive
sudo ./docker/build.sh/ --file docker/ubuntu-18.04.Dockerfile --tag tensorrt-ubuntu18.04-cuda11.4
sudo ./docker/launch.sh --tag tensorrt -ubuntu18.04-cuda11.4 gpus -all --jupyter 8888 
```

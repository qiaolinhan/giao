# NAVLab2024
# qiaolinhan
# 2024-02-06

sudo apt update
sudo apt upgrade -y

# compil opencv from source
# #######################
# # A C/C++ compiler    #
# # cmake               #
# # The Java JDK        #
# # Apache Ant          #
# #######################
#
# to install Java JDK
# ######################
sudo apt install default-jre -y
# sudo apt install openjdk-11-jre-headless -y
# sudo apt install openjdk-17-jre-headless -y
# sudo apt install openjdk-18-jre-headless -y
# sudo apt install openjdk-9-jre-headless -y

java --version

# to install Apache Ant
#  #####################
cd ~/Downloads

# Set environmental variables: JAVA_HOME to your Java environment, ANT_HOME to the directory you uncompressed Ant to, and add ${ANT_HOME}/bin (Unix) or %ANT_HOME%\bin (Windows) to your PATH. See Setup for details.


# to compile opencv
# #######################
wget -c https://github.com/opencv/opencv/archive/4.5.1.tar.gz
tar -xvzf 4.5.1.tar.gz && rm -rf 4.5.1.tar.gz

wget -c https://github.com/opencv/openc_contrib/archive/4.5.1.tar.gz
tar -xvzf 4.5.1.tar.gz && rm -rf 4.5.1.tar.gz

sudo mv opencv-4.5.1/ /opt/opencv4
sudo mv opencv_contrib-4.5.1/ /opt/opencv4_contrib

cd /opt/opencv4
rm -rf build
mkdir build && cd build

# select the path of conda environment
conda_env_path=~/anaconda3/envs/yolov8
opencv_contrib_path=/opt/opencv4_contrib

# specifying OPENCV_ENABLE_NONFREE=ON and conda environment
cmake   -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/opt/opencv/ \
        -D OPENCV_EXTRA_MODULES_PATH=$opencv_contrib_path/modules \
        -D PYTHON3_EXECUTABLE=$conda_env_path/bin/python3.12 \
        -D PYTHON3_LIBRARY=$conda_env_path/lib/python3.12 \
        -D PYTHON3_INCLUDE_DIR=$conda_env_path/include/python3.12 \
        -D PYTHON3_PACKAGES_PATH=$conda_env_path/lib/python3.12/site-packages \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_EXAMPLES=OFF ..
	
# set the number of cores for installation and compile
make -j4
sudo make install

# link the shared library to the conda environment
cd ~/anaconda3/envs/yolov8/lib/python3.12
ln -s /opt/opencv4/build/lib/python3/cv2.cpython-38-x86_64-linux-gnu.so

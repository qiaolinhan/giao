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
sudo apt install default-jdk
sudo apt install default-jre
# sudo apt install openjdk-11-jre-headless -y
# sudo apt install openjdk-17-jre-headless -y
# sudo apt install openjdk-18-jre-headless -y
# sudo apt install openjdk-9-jre-headless -y
java -version
update-alternatives --config java
# which java
# sudo nvim /etc/environment
# add the java path
# (/usr/lib/jvm/java-11-openjdk-amd64/bin/java)
# source /etc/environment
echo $JAVA_HOME

# to install Apache Ant
#  #####################
ant -version

cd ~/Downloads
# wget https://dlcdn.apache.org//ant/binaries/apache-ant-1.10.14-bin.tar.gz
# tar -xvzf apache-ant-1.10.14-bin.tar.gz
# sudo mv apache-ant-1.10.14 ~/apache-ant
 # sudo nvim /etc/environment
# add the java path
# (~/apache-ant/bin)
# source /etc/environment

# Set environmental variables: JAVA_HOME to your Java environment, ANT_HOME to the directory you uncompressed Ant to, and add ${ANT_HOME}/bin (Unix) or %ANT_HOME%\bin (Windows) to your PATH. See Setup for details.

# to compile opencv
# #######################
wget -c https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz
tar -xvzf 4.9.0.tar.gz
sudo mv opencv-4.9.0/ /opt/opencv4
rm -rf 4.9.0.tar.gz

wget -c https://github.com/opencv/opencv_contrib/archive/refs/tags/4.9.0.tar.gz
tar -xvzf 4.9.0.tar.gz
sudo mv opencv_contrib-4.9.0/ /opt/opencv4_contrib
rm -rf 4.9.0.tar.gz


cd /opt/opencv4
rm -rf build
mkdir build && cd build

# select the path of conda environment
conda_env_path=~/anaconda3/envs/dev
opencv_contrib_path=/opt/opencv4_contrib

# specifying OPENCV_ENABLE_NONFREE=ON and conda environment
cmake   -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/opt/opencv/ \
	-D OPENCV_EXTRA_MODULES_PATH=$opencv_contrib_path/modules \
	-D PYTHON3_EXECUTABLE=$conda_env_path/bin/python3.11 \
	-D PYTHON3_LIBRARY=$conda_env_path/lib/libpython3.11.so \
	-D PYTHON3_NUMPY_INCLUDE_DIRS=$conda_env_path/lib/python3.11/site-packages/numpy/core/include \
	-D PYTHON3_PACKAGES_PATH=$conda_env_path/lib/python3.11/site-packages \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_python3=ON \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D BUILD_EXAMPLES=OFF ..
# To check whether there are ANT, JDK, java-wrappers, ...

# set the number of cores for installation and compile
make -j4
sudo make install

# link the shared library to the conda environment
cd ~/anaconda3/envs/dev/lib/python3.11
rm -rf cv2.cpython-38-x86_64-linux-gnu.so
ln -s /opt/opencv4/build/lib/python3/cv2.cpython-311-x86_64-linux-gnu.so

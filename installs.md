# When I Got My T430

## To deply Github
```bash
sudo apt install git
```
define global git
```
git config --global user.name qiaolinhan
git config --global user.email 742954173@qq.com
git config --list
```
 get the ssh-key
```bash
ssh-keygen -t rsa -C "742954173@qq.com"
```

 print the created ssh-key
```bash
cat ~/.ssh/id_rsa.pub
```
add the key on Github </br>
<kbd>settings</kbd> -> <kbd>SSH and GPG keys</kbd> -> <kbd>New SSH key</kbd> -> 
<kbd>Title</kbd> for adding the title of this ssh-key -> Paste the key ->
<kbd>Add SSH key</kbd>

## To deploy Anaconda and Pytorch environment
download the package (On 2022-05-17)
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
```
verify data and integrity with SHA-256
```bash
sha256sum /Anaconda3-xxxxxxxxxxxxxxx.sh
```
install Anaconda3
```bash
bash ~/downloads/Anaconda3-xxxxxxxxxxxxxxxxxx.sh
```
To check, reopen a terminal and type 
```bash
conda --version
```
build a new env and install pytorch
```bash
conda create -n dev
conda activate dev
```
set the environment `dev` as defult
```bash
vi ~/.bashrc
```
add 
```nvim
export PAT="~/anaconda3/envs/dev/bin:$PATH"
conda activate dev
```
install pytorch environment with conda
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
install other nessesary packages
```bash
cd ~/dev
conda env update -f envupdate.yml
```
## To deploy nvim (neo vim)
* install the neo vim support of python
`pip install pynvim` </br>
* install neo vim
`sudo snap install nvim --classic` </br> 
or we could install a release version by:</br>
`wget https://github.com/neovim/neovim/releases/download/v0.7.0/nvim-linux64.deb`</br>
`sudo apt install nvim-linux64.deb` </br>
* install curl on new pc 
`sudo apt install curl` </br>

* install pure_vim
```
cd ~/.config/
git clone https://github.com/lee-shun/pure_vim nvim
```
* some fast words: </br>
  * <kbd>space</kbd> <kbd>f</kbd> <kbd>f</kbd> to find files (ff)
  * <kbd>space</kbd> <kbd>f</kbd> <kbd>m</kbd> to find most recently 
  opened files (fm)
  * <kbd>space</kbd> <kbd>f</kbd> <kbd>b</kbd> to find buffers
  * <kbd>space</kbd> <kbd>t</kbd>  to take a look at the file tree

  * <kbd>ctrl + w</kbd> <kbd>v</kbd> to split new window right (vertically :vsplit)
  * <kbd>ctrl + w</kbd> <kbd>s</kbd> to split new window bottom (horizental :split)
  * <kbd>space</kbd> <kbd>r</kbd><kbd>c</kbd> to set the nvim settings

Translate: put the cursor on the word and type `ts`
Terminal: <F12>
copy(yank): in visual-mode, type 

## connect lazygit and nvim
download the stable release of lazygit
```bash
wget https://github.com/jesseduffield/lazygit/releases/download/v0.34/lazygit_0.34_Linux_x86_64.tar.gz
```
extract lazygit `tar xvf lazygit.tgz` </br>
move it into the default position `sudo mv lazygit /usr/local/bin`

20220728<br>
To easily evaluate the model performance, [`torchmetrics`](https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html) is installed <br>
```
# python package index
pip install torchmetrics
# conda
conda install -c conda-forge torchmetrics
```

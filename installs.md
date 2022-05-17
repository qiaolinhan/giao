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

# get the ssh-key
```bash
ssh-keygen -t rsa -C "742954173@qq.com"
```

# print the created ssh-key
cat ~/.ssh/id_rsa.pub

# add the key on Github
`settings` -> `SSH and GPG keys` -> `New SSH key` -> Name `the Title` -> Paste the key -> `Add SSH key`


##############################
# To deploy Anaconda
##############################
# download the package (On 2022-05-17)
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# verify data and integrity with SHA-256
sha256sum /Anaconda3-xxxxxxxxxxxxxxx.sh

# install Anaconda3
bash ~/downloads/Anaconda3-xxxxxxxxxxxxxxxxxx.sh

# check
reopen a terminal and type 
    conda --version

# build a new env and install pytorch
conda create -n dev
conda activate dev

# set the environment `dev` as defult
vi ~/.bashrc
`i` to insert in vi
add 
    export PATH="~/anaconda3/envs/dev/bin:$PATH"
    conda activate dev
at end


# install pytorch environment with conda
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# install other nessesary packages
cd ~/dev
conda env update -f envupdate.yml

#################################
# To deploy nvim (neo vim)
#################################
# install neo vim
sudo snap install nvim --classic 

#  install curl on new pc 
sudo apt install curl 

# install pure_vim
cd ~/.config/
git clone https://github.com/lee-shun/pure_vim nvim

# some fast words
space + f + f /* find files
space + f + m /* find most recently used
space  f + b /* find buffers
space + t /* take a look at the file tree

$$\sum_{i = 1}^{N}$$
ctrl w + v /* split new window at right (vertically :vsplit)
ctrl w + s /* split new window at bottom (horizental :split)

Translate: put the cursor on the word and type `ts`
Terminal: <F12>
copy(yank): in visual-mode, type 

# some commands
o /* type in next line
i /* insert 
a /* add after cursor(光标)
:q! /* force quit without saving
dd /* delete the entire line
x /* delete the word where the cursor located

# In visual-mode
y /* copy (yank)
yw /* yank this word

# connect lazygit and nvim
    # download the stable release of lazygit
    wget https://github.com/jesseduffield/lazygit/releases/download/v0.34/lazygit_0.34_Linux_x86_64.tar.gz
    
    # extract lazygit
    tar xvf lazygit.tgz

    # move it into the default position
    sudo mv lazygit /usr/local/bin

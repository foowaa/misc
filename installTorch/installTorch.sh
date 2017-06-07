#!/usr/bin/env bash
set -e
cd ~
sudo su
apt-get install git
########################################
#This shell file aims at install torch and the orther deps..
########################################


#############################
#Below install torch deps.
############################
echo 'This will spend almost 1 hrs depending on UR network'
sleep 4s
echo '-------------------Begin-----------------------'
sleep 3s
{

install_openblas() {
    # Get and build OpenBlas (Torch is much better with a decent Blas)
    cd /tmp/
    rm -rf OpenBLAS
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    if [ $(getconf _NPROCESSORS_ONLN) == 1 ]; then
        make NO_AFFINITY=1 USE_OPENMP=0 USE_THREAD=0
    else
        make NO_AFFINITY=1 USE_OPENMP=1
    fi
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be compiled";
        exit $RET;
    fi
    sudo make install
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be installed";
        exit $RET;
    fi
}

install_openblas_AUR() {
    # build and install an OpenBLAS package for Archlinux
    cd /tmp && \
    curl https://aur.archlinux.org/cgit/aur.git/snapshot/openblas-lapack.tar.gz | tar zxf - && \
    cd openblas-lapack
    makepkg -csi --noconfirm
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be installed";
        exit $RET;
    fi
}

checkupdates_archlinux() {
    # checks if archlinux is up to date
    if [[ -n $(checkupdates) ]]; then
        echo "It seems that your system is not up to date."
        echo "It is recommended to update your system before going any further."
        read -p "Continue installation ? [y/N] " yn
            case $yn in
                Y|y ) echo "Continuing...";;
                * ) echo "Installation aborted."
                    echo "Relaunch this script after updating your system with 'pacman -Syu'."
                    exit 0
            esac
    fi
}

# Based on Platform:
if [[ `uname` == 'Darwin' ]]; then
    # GCC?
    if [[ `which gcc` == '' ]]; then
        echo "MacOS doesn't come with GCC: please install XCode and the command line tools."
        exit 1
    fi

    # Install Homebrew (pkg manager):
    if [[ `which brew` == '' ]]; then
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi

    # Install dependencies:
    brew update
    brew install git readline cmake wget qt
    brew install libjpeg imagemagick zeromq graphicsmagick openssl
    brew link readline --force
    brew install caskroom/cask/brew-cask
    brew cask install xquartz
    brew remove gnuplot
    brew install gnuplot --with-wxmac --with-cairo --with-pdflib-lite --with-x11 --without-lua

elif [[ "$(uname)" == 'Linux' ]]; then

    if [[ -r /etc/os-release ]]; then
        # this will get the required information without dirtying any env state
        DIST_VERS="$( ( . /etc/os-release &>/dev/null
                        echo "$ID $VERSION_ID") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    elif [[ -r /etc/redhat-release ]]; then
        DIST_VERS=( $( cat /etc/redhat-release ) ) # make the file an array
        DISTRO="${DIST_VERS[0],,}" # get the first element and get lcase
        VERS="${DIST_VERS[2]}" # get the third element (version)
    elif [[ -r /etc/lsb-release ]]; then
        DIST_VERS="$( ( . /etc/lsb-release &>/dev/null
                        echo "${DISTRIB_ID,,} $DISTRIB_RELEASE") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    else # well, I'm out of ideas for now
        echo '==> Failed to determine distro and version.'
        exit 1
    fi

    # Detect fedora
    if [[ "$DISTRO" = "fedora" ]]; then
        distribution="fedora"
        fedora_major_version="$VERSION"
    # Detect archlinux
    elif [[ "$DISTRO" = "arch" ]]; then
        distribution="archlinux"
    # Detect Ubuntu
    elif [[ "$DISTRO" = "ubuntu" ]]; then
        export DEBIAN_FRONTEND=noninteractive
        distribution="ubuntu"
        ubuntu_major_version="${VERSION%%.*}"
    # Detect elementary OS
    elif [[ "$DISTRO" = "elementary" ]]; then
        export DEBIAN_FRONTEND=noninteractive
        distribution="elementary"
        elementary_version="${VERSION%.*}"
    # Detect CentOS
    elif [[ "$DISTRO" = "centos" ]]; then
        distribution="centos"
        centos_major_version="$VERSION"
    # Detect AWS
    elif [[ "$DISTRO" = "amzn" ]]; then
        distribution="amzn"
        amzn_major_version="$VERSION"
    else
        echo '==> Only Ubuntu, elementary OS, Fedora, Archlinux and CentOS distributions are supported.'
        exit 1
    fi

    # Install dependencies for Torch:
    if [[ $distribution == 'ubuntu' ]]; then
        sudo apt-get update
        # python-software-properties is required for apt-add-repository
        sudo apt-get install -y python-software-properties
        echo "==> Found Ubuntu version ${ubuntu_major_version}.xx"
        if [[ $ubuntu_major_version -lt '12' ]]; then
            echo '==> Ubuntu version not supported.'
            exit 1
        elif [[ $ubuntu_major_version -lt '14' ]]; then
            sudo add-apt-repository -y ppa:chris-lea/zeromq
            sudo add-apt-repository -y ppa:chris-lea/node.js
        elif [[ $ubuntu_major_version -lt '15' ]]; then
            sudo add-apt-repository -y ppa:jtaylor/ipython
        else
            sudo apt-get install -y software-properties-common \
                libgraphicsmagick1-dev nodejs npm libfftw3-dev sox libsox-dev \
                libsox-fmt-all
        fi

        gcc_major_version=$(gcc --version | grep ^gcc | awk '{print $4}' | \
                            cut -c 1)
        if [[ $gcc_major_version == '5' ]]; then
            echo '==> Found GCC 5, installing GCC 4.9.'
            sudo apt-get install -y gcc-4.9 libgfortran-4.9-dev g++-4.9
        fi

        sudo apt-get update
        sudo apt-get install -y build-essential gcc g++ curl \
            cmake libreadline-dev git-core libqt4-core libqt4-gui \
            libqt4-dev libjpeg-dev libpng-dev ncurses-dev \
            imagemagick libzmq3-dev gfortran unzip gnuplot \
            gnuplot-x11 ipython

        if [[ $ubuntu_major_version -lt '14' ]]; then
            # Install from source after installing git and build-essential
            install_openblas || true
        else
            sudo apt-get install -y libopenblas-dev liblapack-dev
        fi

    elif [[ $distribution == 'elementary' ]]; then
        declare -a target_pkgs
        target_pkgs=( build-essential gcc g++ curl \
                      cmake libreadline-dev git-core libqt4-core libqt4-gui \
                      libqt4-dev libjpeg-dev libpng-dev ncurses-dev \
                      imagemagick libzmq3-dev gfortran unzip gnuplot \
                      gnuplot-x11 ipython )
        sudo apt-get update
        # python-software-properties is required for apt-add-repository
        sudo apt-get install -y python-software-properties
        if [[ $elementary_version == '0.3' ]]; then
            echo '==> Found Ubuntu version 14.xx based elementary installation, installing dependencies'
            sudo apt-get install -y software-properties-common \
                libgraphicsmagick1-dev nodejs npm libfftw3-dev sox libsox-dev \
                libsox-fmt-all

            sudo add-apt-repository -y ppa:jtaylor/ipython
        else
            sudo add-apt-repository -y ppa:chris-lea/zeromq
            sudo add-apt-repository -y ppa:chris-lea/node.js
        fi
        sudo apt-get update
        sudo apt-get install -y "${target_pkgs[@]}"

        install_openblas || true

    elif [[ $distribution == 'archlinux' ]]; then
        echo "Archlinux installation"
        checkupdates_archlinux
        sudo pacman -S --quiet --noconfirm --needed \
            cmake curl readline ncurses git \
            gnuplot unzip libjpeg-turbo libpng libpng \
            imagemagick graphicsmagick fftw sox zeromq \
            ipython qt4 qtwebkit || exit 1
        pacman -Sl multilib &>/dev/null
        if [[ $? -ne 0 ]]; then
            gcc_package="gcc"
        else
            gcc_package="gcc-multilib"
        fi
        sudo pacman -S --quiet --noconfirm --needed \
            ${gcc_package} gcc-fortran || exit 1
        # if openblas is not installed yet
        pacman -Qs openblas &> /dev/null
        if [[ $? -ne 0 ]]; then
            install_openblas_AUR || true
        else
            echo "OpenBLAS is already installed"
        fi

    elif [[ $distribution == 'fedora' ]]; then
        if [[ $fedora_major_version == '20' ]]; then
            sudo yum install -y cmake curl readline-devel ncurses-devel \
                                gcc-c++ gcc-gfortran git gnuplot unzip \
                                nodejs npm libjpeg-turbo-devel libpng-devel \
                                ImageMagick GraphicsMagick-devel fftw-devel \
                                sox-devel sox zeromq3-devel \
                                qt-devel qtwebkit-devel sox-plugins-freeworld \
                                ipython
            install_openblas || true
        elif [[ $fedora_major_version == '22' ||  $fedora_major_version == '23'  ]]; then
            #using dnf - since yum has been deprecated
            #sox-plugins-freeworld is not yet available in repos for F22
            sudo dnf install -y cmake curl readline-devel ncurses-devel \
            			gcc-c++ gcc-gfortran git gnuplot unzip \
            			nodejs npm libjpeg-turbo-devel libpng-devel \
            			ImageMagick GraphicsMagick-devel fftw-devel \
            			sox-devel sox qt-devel qtwebkit-devel \
            			python-ipython czmq czmq-devel
            install_openblas || true
        else
            echo "Only Fedora 20 or Fedora 22 is supported for now, aborting."
            exit 1
        fi
    elif [[ $distribution == 'centos' ]]; then
        if [[ $centos_major_version == '7' ]]; then
            sudo yum install -y epel-release # a lot of things live in EPEL
            sudo yum install -y cmake curl readline-devel ncurses-devel \
                                gcc-c++ gcc-gfortran git gnuplot unzip \
                                nodejs npm libjpeg-turbo-devel libpng-devel \
                                ImageMagick GraphicsMagick-devel fftw-devel \
                                sox-devel sox zeromq3-devel \
                                qt-devel qtwebkit-devel sox-plugins-freeworld
            sudo yum install -y python-ipython
            install_openblas || true
        else
            echo "Only CentOS 7 is supported for now, aborting."
            exit 1
        fi
    elif [[ $distribution == 'amzn' ]]; then
        sudo yum install -y cmake curl readline-devel ncurses-devel \
                            gcc-c++ gcc-gfortran git gnuplot unzip \
                            nodejs npm libjpeg-turbo-devel libpng-devel \
                            ImageMagick GraphicsMagick-devel fftw-devel \
                            sox-devel sox zeromq3-devel \
                            qt-devel qtwebkit-devel sox-plugins-freeworld \
                            ipython libgfortran
        install_openblas || true
    fi

else
    # Unsupported
    echo '==> platform not supported, aborting'
    exit 1
fi

ipython_exists=$(command -v ipython)
if [[ $ipython_exists ]]; then {
    ipython_version=$(ipython --version|cut -f1 -d'.')
    if [[ $ipython_version != 2 && $ipython_version != 3 && $ipython_version != 4 ]]; then {
        echo 'WARNING: Your ipython version is too old.  Type "ipython --version" to see this.  Should be at least version 2'
    } fi
} fi

# Done.
echo "==> Torch7's dependencies have been installed"

}
sleep 3s

############################################
#Below install Torch
###########################################
echo 'This will spend almost 0.5 hrs depending on UR network'
sleep 4s
echo '-------------------------begin-----------------------'
sleep 3s


{

# Prefix:
PREFIX=${PREFIX-/usr/local}
echo "Installing Torch into: $PREFIX"

if [[ `uname` == 'Linux' ]]; then
    export CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include:/opt/OpenBLAS/lib:$CMAKE_LIBRARY_PATH
fi

# Build and install Torch7
cd /tmp
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir build; cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install || sudo -E make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# check if we are on mac and fix RPATH for local install
path_to_install_name_tool=$(which install_name_tool)
if [ -x "$path_to_install_name_tool" ] 
then
   install_name_tool -id ${PREFIX}/lib/libluajit.dylib ${PREFIX}/lib/libluajit.dylib
fi

# Statuses:
sundown=ok
cwrap=ok
paths=ok
torch=ok
nn=ok
dok=ok
gnuplot=ok
qtlua=ok
qttorch=ok
lfs=ok
penlight=ok
sys=ok
xlua=ok
image=ok
optim=ok
cjson=ok
trepl=ok

path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then  
    cutorch=ok
    cunn=ok
fi

# Install base packages:
$PREFIX/bin/luarocks install sundown       ||  sudo -E $PREFIX/bin/luarocks install sundown
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install cwrap         ||  sudo -E $PREFIX/bin/luarocks install cwrap  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install paths         ||  sudo -E $PREFIX/bin/luarocks install paths  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install torch         ||  sudo -E $PREFIX/bin/luarocks install torch  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install nn            ||  sudo -E $PREFIX/bin/luarocks install nn     
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install dok           ||  sudo -E $PREFIX/bin/luarocks install dok    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install gnuplot       ||  sudo -E $PREFIX/bin/luarocks install gnuplot
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
[ -n "$cutorch" ] && \
($PREFIX/bin/luarocks install cutorch      ||  sudo -E $PREFIX/bin/luarocks install cutorch        ||   cutorch=failed )
[ -n "$cunn" ] && \
($PREFIX/bin/luarocks install cunn         ||  sudo -E $PREFIX/bin/luarocks install cunn           ||   cunn=failed )

$PREFIX/bin/luarocks install qtlua         ||  sudo -E $PREFIX/bin/luarocks install qtlua  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install qttorch       ||  sudo -E $PREFIX/bin/luarocks install qttorch
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install luafilesystem ||  sudo -E $PREFIX/bin/luarocks install luafilesystem
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install penlight      ||  sudo -E $PREFIX/bin/luarocks install penlight 
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install sys           ||  sudo -E $PREFIX/bin/luarocks install sys      
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install xlua          ||  sudo -E $PREFIX/bin/luarocks install xlua     
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install image         ||  sudo -E $PREFIX/bin/luarocks install image    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install optim         ||  sudo -E $PREFIX/bin/luarocks install optim    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install lua-cjson     ||  sudo -E $PREFIX/bin/luarocks install lua-cjson
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install trepl         ||  sudo -E $PREFIX/bin/luarocks install trepl    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi

# Done.
echo ""
echo "=> Torch7 has been installed successfully"
echo ""
echo "  + Extra packages have been installed as well:"
echo "     $ luarocks list"
echo ""
echo "  + To install more packages, do:"
echo "     $ luarocks search --all"
echo "     $ luarocks install PKG_NAME"
echo ""
echo "  + Note: on MacOS, it's a good idea to install GCC 5 to enable OpenMP."
echo "     You can do this by with brew"
echo "      $ brew install gcc --without-multilib"
echo "     type the following lines before running the installation script"
echo "      export CC=gcc-5"
echo "      export CXX=g++-5"
echo "     For installing cunn, you will need instead the default AppleClang compiler,"
echo "     which means you should open a new terminal (with unexported CC and CXX) and"
echo "      luarocks install cunn"
echo ""
echo "  + packages installed:"
echo "    - sundown   : " $sundown
echo "    - cwrap     : " $cwrap
echo "    - paths     : " $paths
echo "    - torch     : " $torch
echo "    - nn        : " $nn
echo "    - dok       : " $dok
echo "    - gnuplot   : " $gnuplot
[ -n "$cutorch" ] && echo "    - cutorch   : " $cutorch
[ -n "$cunn" ]    && echo "    - cunn      : " $cunn
echo "    - qtlua     : " $qtlua
echo "    - qttorch   : " $qttorch
echo "    - lfs       : " $lfs
echo "    - penlight  : " $penlight
echo "    - sys       : " $sys
echo "    - xlua      : " $xlua
echo "    - image     : " $image
echo "    - optim     : " $optim
echo "    - cjson     : " $cjson
echo "    - trepl     : " $trepl
echo ""

}

#######################################
#Try installing iTorch
#######################################


echo 'IF USING CUDA, PLZ INSTALL THEM MANUALLY!'
echo 'AFTER INSTALLATION, luarocks install cutorch AND luarocks install cunn'
sleep 4s
echo 'Try installing iTorch, I have not succeeded installing it, actually lol'
sleep 5s
echo '--------------------begin-----------------------'
sleep 3s

apt-get install curl
apt-get install aptitude
curl -O https://archive.org/download/zeromq_4.0.5/zeromq-4.0.5.tar.gz
tar -xf zeromq-4.0.5.tar.gz
cd ./zeromq-4.0.5
./configure
make
make install 
cd~

aptitude install python-dev
aptitude install python-pip
pip install ipython
pip install jupyter
pip install pyzmq
pip install MarkupSafe
pip install jsonschema
pip install jinja2
pip install tornado

aptitude install --upgrade openssl
aptitude install --upgrade libssl

luarocks install luacrypto
luarocks install env
luarocks install lzmq
luarocks install lbase64
luarocks install uuid
aptitude install gstreamer1.0-libav 
# Install iTorch
git clone https://github.com/facebook/iTorch.git
cd iTorch
sudo env "PATH=$PATH" luarocks make
sudo chown -R $USER $(dirname $(ipython locate profile))

echo '-------------------------------------------'
echo 'itorch AND itorch notebook to confirm'
echo '-------------------------------------------'

sleep 5s

#############################################
#INSTALL ZBStudio
#############################################
echo 'installing ZBStudio'
sleep 3s
echo '------------------begin---------------------'
sleep 3s
cd ~
curl -o zbs.sh https://download.zerobrane.com/ZeroBraneStudioEduPack-1.40-linux.sh
sh ./zbs.sh
cd /opt/zbstudio/cfg/
touch user.lua
comm='path.lua=/usr/local/bin/qlua'
echo $comm>>user.lua
luarocks install qtlua
luarocks install qttorch
echo 'use zbstudio to start this IDE'
sleep 3s


###############################################
#INSTALL SOME USEFUL PACKAGES
###############################################
luarocks install nngraph
luarocks install hdf5
luarocks install csvigo
cd ~
git clone https://github.com/twitter/torch-autograd
cd ./torch-autograd
luarocks make
luarocks install cudnn

echo "ALL DOWN"

FROM tensorflow/haskell:v0

RUN \
    apt-get update && \
    apt-get install -y \
       cmake \
       git \
       libsdl-gfx1.2-dev \
       libsdl-image1.2-dev \
       libsdl1.2-dev

RUN \
    git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git /ale && \
    cd /ale && \
    mkdir build && cd build && \
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .. && \
    make -j 4 && \
    install ../ale_python_interface/libale_c.so /usr/local/lib

RUN \
    apt-get install -y wget && \
    mkdir /sdl_scratch && \
    cd /sdl_scratch && \
    wget https://www.libsdl.org/release/SDL2-2.0.5.tar.gz && \
    tar xvzf SDL2-2.0.5.tar.gz && \
    cd SDL2-2.0.5 && \
    ./configure && make -j4 && make install

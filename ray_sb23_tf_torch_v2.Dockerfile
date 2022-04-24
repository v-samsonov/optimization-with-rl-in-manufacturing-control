ARG GPU=-cpu
FROM rayproject/ray:nightly"$GPU"

ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.7.7

ARG TENSORFLOW_TYPE=tensorflow-base
ARG TENSORFLOW_VERSION=1.14.0
ARG TENSORFLOW_CHANEL=conda-forge

USER root
#
ENV TZ=Europe/Amsterdam
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
Run sudo echo $TZ > /etc/timezone
#
#USER 1000 # RAY_UID
USER ray
#ENV HOME=/home/ray

RUN sudo apt-get update \
    && sudo apt-get install -y gcc \
       cmake \
       default-jre \
       swig \
       git \
       wget\
       jq \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libfontconfig1 \
       libxrender1 \
       libopenmpi-dev \
       zlib1g-dev \
       ffmpeg \
       freeglut3-dev \
       xvfb \
       libgl1-mesa-glx


# Install anaconda dependencies
RUN \
     $HOME/anaconda3/bin/conda install -y $TENSORFLOW_TYPE=$TENSORFLOW_VERSION -c $TENSORFLOW_CHANEL && \
	 $HOME/anaconda3/bin/conda install -y pytorch $PYTORCH_DEPS -c pytorch && \
	 $HOME/anaconda3/bin/conda install -y gast -c anaconda

# ENV PATH /opt/conda/bin:$PATH

# ENV CODE_DIR /root/code

RUN \
    $HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir uninstall -y opencv-python && \
    $HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install opencv-python-headless && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install wandb && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install ortools==7.4.7247 && \
    $HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install simpy==4.0.1 && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install recordtype==1.3 && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install Pillow==7.2.0 && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install stable-baselines==2.10.1 && \
	$HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install stable-baselines3 && \
    $HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install plotly==4.12.0 && \
    $HOME/anaconda3/bin/pip --use-deprecated=legacy-resolver --no-cache-dir install google-pasta

CMD /bin/bash
RUN \
    mkdir rl4jsp && \
    cd rl4jsp

COPY --chown=ray:users . ./rl4jsp
WORKDIR rl4jsp/RL_Code/scripts
# ENTRYPOINT ["python", "from pathlib import Path; pt = Path.cwd(); files = [f for f in pt.iterdir()]; print(files)"]
# ENTRYPOINT ["python", "run_experiment_test.py"]
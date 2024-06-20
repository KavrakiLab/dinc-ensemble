FROM mambaorg/micromamba

USER root
# Some linux dependencies 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

USER $MAMBA_USER
# Copy needed files to the environment
COPY --chown=$MAMBA_USER:$MAMBA_USER  ./environment.yml .
# Install most of the necessary stuff from the yml file
RUN micromamba create -y -n dinc-ensemble -f environment.yml && \
     micromamba clean --all --yes

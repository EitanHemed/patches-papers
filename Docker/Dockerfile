FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
RUN git clone https://github.com/EitanHemed/patches-papers

RUN conda update -n base -c defaults conda

RUN conda create --name patches python=3.9.12 pip jupyter matplotlib=3.6.2 seaborn=0.12.2 pandas=1.2.5 r-base=4.1.3 gcc r-broom r-afex -y --quiet -c conda-forge

RUN echo "conda activate patches" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy notebook first
COPY reproduce.ipynb .

RUN apt-get update && apt-get install -y \
    fontconfig \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Move notebook and install the package in the patches environment
RUN mv reproduce.ipynb "patches-papers/Code" && \
    cd "patches-papers/Code" && \
    conda run -n patches pip install -e . && \
    rm -r -f po_utils.egg-info && \
    cd ../..
RUN conda run -n patches python -c "import robusta as rst"

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint.sh executable
RUN chmod +x /app/entrypoint.sh

# Expose the notebook port
EXPOSE 8888

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
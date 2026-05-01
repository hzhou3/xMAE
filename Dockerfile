FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# # Upgrade pip to the latest version
# RUN pip install --upgrade pip

# Install all Python dependencies in a single layer.
RUN pip install --no-cache-dir \
    # Core ML and Data Libraries
    numpy \
    pandas \
    scikit_learn \
    pytorch_lightning \
    torchmetrics \
    torch-optimizer \
    torchvision \
    torchaudio \
    # AWS and S3
    boto3 \
    s3fs \
    # Data Formats and I/O
    h5py \
    pyarrow \
    PyYAML \
    # Visualization and Utilities
    matplotlib \
    seaborn \
    ipython \
    jupyter \
    tqdm \
    joblib \
    tabulate \
    typer \
    # Domain-Specific Libraries
    neurokit2 \
    peakutils \
    wandb \
    peft

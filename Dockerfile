FROM condaforge/miniforge3:latest

WORKDIR /app

RUN conda install -y -c conda-forge \
    python=3.13 \
    git \
    xarray=2024.11.0 \
    numpy \
    dask \
    xesmf \
    cftime \
    pyyaml \
    click \
    pandas \
    tqdm \
    netcdf4 \
    h5netcdf \
    cmor \
    && conda clean -afy

# CMOR tables live in a separate repo; vendor them so the image is self-contained.
# (When the project root is bind-mounted over /app at runtime, the host must also
# provide cmip7-cmor-tables/ since the mount hides this copy.)
RUN git clone --depth 1 --branch cesm-dev \
    https://github.com/CESM-Development/cmip7-cmor-tables.git \
    /app/cmip7-cmor-tables

RUN pip install --no-cache-dir \
    dulwich \
    geocat-comp \
    CMIP7-data-request-api

COPY . .

RUN pip install -e .

# Mount the project root at runtime to pick up source edits:
#   docker run --rm -v $(pwd):/app <image> <command>

# syntax=docker/dockerfile:1
#
# cmip7-prep container image.
#
FROM condaforge/miniforge3:26.5.3-0

RUN conda create -y -p /opt/env -c conda-forge \
        python=3.12 \
        cmor=3.15 \
        esmpy \
        xesmf \
        geocat-comp \
        xarray \
        numpy \
        pandas \
        dask \
        netcdf4 \
        h5netcdf \
        cftime \
        pyyaml \
        click \
        matplotlib-base \
        pytest \
        pytest-cov \
        git \
    && conda clean --all --force-pkgs-dirs --yes

ENV PATH=/opt/env/bin:$PATH

RUN pip install --no-cache-dir \
        gents \
        dulwich \
        cmip7-data-request-api

WORKDIR /opt/cmip7-prep
COPY . .

ARG TABLES_REPO=https://github.com/CESM-Development/cmip7-cmor-tables.git
ARG TABLES_REF=cesm-dev
RUN rm -rf cmip7-cmor-tables \
    && git clone --depth 1 --branch "${TABLES_REF}" "${TABLES_REPO}" cmip7-cmor-tables

RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/opt/cmip7-prep

CMD ["bash"]

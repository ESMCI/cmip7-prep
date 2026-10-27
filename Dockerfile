# syntax=docker/dockerfile:1
#
# cmip7-prep container image.
#
FROM condaforge/miniforge3:26.5.3-0

RUN conda install -y -n base -c conda-forge \
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
        jsonschema \
        click \
        matplotlib-base \
        pytest \
        pytest-cov \
        git \
    && conda clean --all --force-pkgs-dirs --yes

RUN pip install --no-cache-dir \
        gents \
        dulwich \
        cmip7-data-request-api==1.4

WORKDIR /opt/cmip7-prep
COPY . .

RUN test -f cmip7-cmor-tables/tables-cvs/cmor-cvs.json

RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/opt/cmip7-prep

CMD ["bash"]

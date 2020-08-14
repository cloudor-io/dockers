FROM continuumio/anaconda3:2020.07

RUN conda install -y -c anaconda cudatoolkit && \
        pip install numba && \
        conda clean -a

COPY ./examples /opt/examples

CMD python /opt/examples/numba_device_query.py

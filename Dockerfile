FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

COPY requirements_verbose.txt .
RUN pip install -r requirements_verbose.txt

ENTRYPOINT ["python", "main.py"]

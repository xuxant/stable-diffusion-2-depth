# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

#ARG GDRIVE_ID
# Install git
RUN apt-get update && apt-get install -y git wget

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1d8tufvV_6f3Zb93jSsQomw_rsD4-qr7x' -O emb.pt

# Add your custom app code, init() and inference()

EXPOSE 8000

ADD user_src.py .
CMD python3 -u server.py

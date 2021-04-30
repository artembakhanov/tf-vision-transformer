FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

COPY . .


RUN apt update && apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

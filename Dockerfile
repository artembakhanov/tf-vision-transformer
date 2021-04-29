FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

COPY . .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
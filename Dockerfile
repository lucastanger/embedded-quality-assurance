# Use the official TensorFlow GPU image with Jupyter support
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Set the working directory in the container
WORKDIR /tf

# Expose the Jupyter notebook port
EXPOSE 8888

# Copy any additional content you want into the image (optional)
# COPY . /tf

# Set up a Jupyter notebook configuration
RUN mkdir -p /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python pandas scikit-learn scipy tflite_runtime seaborn matplotlib

# Command to run Jupyter notebook server
CMD ["jupyter", "notebook", "--allow-root"]

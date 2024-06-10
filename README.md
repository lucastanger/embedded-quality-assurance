# Quality Assurance - An embedded machine learning approach 
A TFLite Model Evaluation for embedded quality assurance

![image](https://github.com/lucastanger/embedded-quality-assurance/assets/39199539/f35f70d7-03b6-4b3a-baa8-afb05bf6cf4f)

# Connect Docker TensorFlow Runtime

To run the Notebooks contained in this repository, I highly recommend you to use the provided Dockerfile. The recommended setup is WSL with a NVIDIA GPU.

Simply run these commands to get started:

````bash
$ docker build -t tf-gpu-jupyter .
````

To launch the Docker Container

````bash
$ docker run -it --rm -p 8888:8888 --gpus all -v $(pwd):/tf tf-gpu-jupyter 
````
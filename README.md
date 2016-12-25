

# Deep Learning for sentiment analysis 

The sentiment predictor is built with a Convolutional Neural Network model realized by Keras API running Tensorflow as backend. The feature embedding is using pretrained sentiment140 model.
 
1. Required packages

| Package    | Version   | Installation |
|:----------:|:---------:|:------------:|
| keras      | 1.0.3     |    PIP       |
| theano     | 0.8.2     |    PIP       |
| tensorflow | 0.12.0rc0 |    PIP       |
| pandas     | 0.19.1    |    PIP       |
| sklearn    | 0.08.1    |    PIP       |
| flask      | 0.11.1    |    PIP       |
| tweepy     | 3.5.0     |    PIP       |
| h5py       | 2.6.0     |    PIP       |

2. Installation script for deep learning modules

```bash
pip install keras==1.0.3       
pip install theano==0.8.2       
pip install tensorflow==0.12.0rc0       
pip install pandas==0.19.1       
pip install sklearn==0.08.1       
pip install flask==0.11.1       
pip install tweep==3.5.0       
pip install h5py==2.6.0
```

# Web service 

Web service is built with Python Flask.

# Deploy to Heroku 

1. Install virtual environment
```bash
sudo python install virtualenv
``` 
1. Set up a new virtual environment with name _venv_
```bash
virtualenv venv
```
1. Activate the virtual environment
```bash
souce ./venv/Scripts/activate
```
1. Install all requirement Python packages
```bash
pip install keras==1.0.3       
pip install theano==0.8.2       
pip install tensorflow==0.12.0rc0       
pip install pandas==0.19.1       
pip install sklearn==0.08.1       
pip install flask==0.11.1       
pip install tweep==3.5.0       
pip install h5py==2.6.0
```
1. Create a dependency file _requirement.txt_ which include all packages and patterns. We do this via
```bash
pip freeze > requirement.txt
```
1. Tensorflow needs some special treatment (revision) to the requirement file. So remove the tensor flow line, something like
```bash
tensorflow==0.10.0
```
and add one line
```bash
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
```
1. Create a _runtime.txt_ file and add the following line to declare python version used in this web app
```bash
python-2.7.12
```
1. Create a _Procfile_ file and add the following line to specify how to run the application when deployed
```bash
web: bin/web
```
also create the _bin/web_ file with the following content
```bash
python app.py
```
1. Version control via Git all required files. 
1. Push to Heroku repository 
```bash
git push -u heroku master
```








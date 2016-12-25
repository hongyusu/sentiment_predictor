
# Required Packages

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

# Installation script

'''bash
pip install keras==1.0.3       
pip install theano==0.8.2       
pip install tensorflow==0.12.0rc0       
pip install pandas==0.19.1       
pip install sklearn==0.08.1       
pip install flask==0.11.1       
pip install tweep==3.5.0       
pip install h5py==2.6.0
'''

# Heroku

1. Version control the following files with Git
```bash
.gitignore
Procfile
README.md
app.py
bin/web
imdb-word-index-map.pickle
model_cnn_sentiment.h5
model_cnn_sentiment.json
requirements.txt
runtime.txt
sentiment_predictor.py
static/airports.json
static/app.js
static/display.js
static/example_trips.json
static/example_trips_short.json
static/images/airplane68.png
static/images/facebook.png
static/images/flight.svg
static/images/hotel.svg
static/images/international.png
static/images/loader.gif
static/images/plate.png
static/images/site.png
static/images/site.svg
static/images/two209.png
static/partner.js
static/util.js
templates/index.html
templates/layout.html
templates/modal.html
templates/results.html
test.txt
wrapper_twitter.py
```

```bash
git push -u heroku master
```








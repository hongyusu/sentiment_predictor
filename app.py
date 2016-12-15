

import os
import json
import flask
from flask import Flask
from flask import render_template, request
import cPickle

app = Flask(__name__)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control']   = 'public, max-age=0'
    return response


@app.route('/')
def index(name=None):
    return render_template('index.html')


@app.route('/action1', methods=['POST'])
def action1(name=None):
    ht = request.get_json()
    cPickle.dump(ht,open("hashtag.pickle","wb"))
    print "---> {} : started".format(ht)
    #os.system("python sentiment_predictor.py")
    while True:
        if os.path.isfile("hashtag_res.pickle"):
            try:
                data = cPickle.load(open("hashtag_res.pickle","r+"))
                os.system("rm hashtag_res.pickle")
                print "---> {} : ended".format(ht)
                break
            except:
                pass
    return flask.jsonify(data)


#app.debug = True 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    os.system("rm hashtag*; python wrapper_twitter.py &")
    app.run(host='0.0.0.0', port=port)



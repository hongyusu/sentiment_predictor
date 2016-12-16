


import os
import tweepy
import sys
import sentiment_predictor
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops 
import cPickle
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adadelta


import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

tf.python.control_flow_ops = control_flow_ops

CONSUMER_KEY = 'ncMZ2CP7YmScHkLYwmfCYaTZz'
CONSUMER_SECRET = 'ZkFEJXxXEOUlqkhrJ14kzWakrXjqIe11de7ks28DyC79P31t9q'
ACCESS_KEY = '1157786504-XB3DXGrMmhvM1PAb6aeys3LJFYI9Y3LzS6veRHj'
ACCESS_SECRET = '8w69uDRm9PPA9iv3fNtkHPKP4FIq5SFtVbcE28wtcY5qx'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)




def get_tweets_given_hashtag(hashtag,amount=30):
    '''
    get tweets by hashtag
    '''
    tweets = api.search(hashtag, count=amount)
    tweets = [tweet.text for tweet in tweets]
    return tweets

def predict_given_tweets(tweets):
    '''
    make prediction for a collection of tweets
    '''
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # readin index
        word_index_map = cPickle.load(open("imdb-word-index-map.pickle", "rb"))

        # load model and parameters from file
        with open('model_cnn_sentiment.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model_cnn_sentiment.h5")
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

        # make prediction    
        scores = sentiment_predictor.predict_given_sentences(tweets,word_index_map,model)

        return scores




def operate_on_hashtag_cli(hashtag,amount=30):
    """
    fetch a collection of tweets based on hashtag from file for which predictions are made 
    """
    tweets = get_tweets_given_hashtag(hashtag)
    scores = predict_given_tweets(tweets)
    
    i = 0
    for score in scores:
        print score,tweets[i]
        i+=1

    res = {}
    if tweets:
        res['status']    = 0
        res['items']     = tweets
        res['scores']    = scores
        res['meanscore'] = sum(scores)/len(scores)

    return res

def operate_on_hashtag_file(hashtag_picklefilename,amount=30):
    """
    fetch a collection of tweets based on hashtag from CLI for which predictions are made 
    """
    try:
        hashtag = cPickle.load(open(hashtag_picklefilename,"rb"))
    except:
        with open(hashtag_picklefilename) as f:
            hashtag = f.read()
    tweets = get_tweets_given_hashtag(hashtag,amount)
    scores = predict_given_tweets(tweets)
    
    i = 0
    for score in scores:
        print score,tweets[i]
        i+=1

    res = {}
    if tweets:
        res['status']    = 0
        res['items']     = tweets
        res['scores']    = scores
        res['meanscore'] = sum(scores)/len(scores)

    return res


def monitor_hashtag_pickle_files():
    while True:
        if os.path.isfile("hashtag.pickle"):
            try:
                hashtag = cPickle.load(open("hashtag.pickle","rb"))
                print "---> {} : loaded".format(hashtag)
                os.system("rm hashtag.pickle")
                tweets = get_tweets_given_hashtag(hashtag,10)
                print tweets
                scores = predict_given_tweets(tweets)
                scores = scores[:,1].tolist()
                print "---> {} : scored".format(hashtag)


                res = {}
                if tweets:
                    res['status'] = 0
                    res['items'] = tweets
                    res['scores'] = scores
                    res['meanscore'] = sum(scores)/len(scores)

                cPickle.dump(res,open("hashtag_res.pickle","wb"))
            except:
                pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        monitor_hashtag_pickle_files()
    else:
        operate_on_hashtag_cli(sys.argv[1])
        #operate_on_hashtag_file(sys.argv[1])


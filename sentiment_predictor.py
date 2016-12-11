


import numpy as np
import pandas as pd
import cPickle
import re
from collections import defaultdict
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.constraints import unitnorm
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score


import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def generate_data_train_test(data_train, f1, f2, data_test, f3, train_ratio = 0.8, clean_string=True):
    """
    generate data for training (training/test) and test
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set
    trainingsize = data_train.shape[0]  
    trainingsize = 100
    for i in xrange(trainingsize):
        line = data_train[f1][i]
        y    = data_train[f2][i]
        rev  = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {'y': y, 
                  'text': orig_rev,
                  'num_words': len(orig_rev.split()),
                  'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)
        
    # Pre-process test data set
    testsize = data_test.shape[0]  
    testsize = 100
    for i in xrange(testsize):
        line = data_test[f3][i]
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {'y': -1, 
                  'text': orig_rev,
                  'num_words': len(orig_rev.split()),
                  'split': -1}
        revs.append(datum)
        
    return revs, vocab

def load_google_w2v(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

    return word_vecs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
    W[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map



def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l, kernel_size)
        sent.append(rev['y'])
        if rev['split'] == 1:
            train.append(sent)
        elif rev['split'] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=np.int)
    val = np.array(val, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return [train, val, test]

def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x



def preprocessing():
    # Read and load data
    data_train  = pd.read_csv('DATA/labeledTrainData.tsv', sep='\t')
    data_test   = pd.read_csv('DATA/testData.tsv', sep='\t')
    revs, vocab = generate_data_train_test(data_train, "review", "sentiment", data_test, "review", train_ratio=0.8, clean_string=True)

    max_l = np.max(pd.DataFrame(revs)['num_words'])
    print 'data loaded!'
    print 'number of sentences: ' + str(len(revs))
    print 'vocab size: ' + str(len(vocab))
    print 'max sentence length: ' + str(max_l)
    print 'loading word2vec vectors...',
    
    # Load Google w2v file
    w2v_file   = '/Users/hongyusu/Data/GoogleNews-vectors-negative300.bin'
    w2v = load_google_w2v(w2v_file, vocab)

    print 'word2vec loaded!'
    print 'num words already in word2vec: ' + str(len(w2v))


    # add unknown word
    w2v = add_unknown_words(w2v, vocab) 
    W, word_idx_map = get_W(w2v)

    # save dataset
    cPickle.dump([revs, W, word_idx_map, vocab], open('imdb-train-val-test.pickle', 'wb'))
    print 'dataset created!'




def learning():

    print "loading data..."
    x = cPickle.load(open("imdb-train-val-test.pickle", "rb"))
    revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    print "data loaded!"
    datasets = make_idx_data(revs, word_idx_map, max_l=2633, kernel_size=5)

    # Train data preparation
    N = datasets[0].shape[0]
    conv_input_width = W.shape[1]
    conv_input_height = int(datasets[0].shape[1]-1)

    # For each word write a word index (not vector) to X tensor
    train_X = np.zeros((N, conv_input_height), dtype=np.int)
    train_Y = np.zeros((N, 2), dtype=np.int)
    for i in xrange(N):
        for j in xrange(conv_input_height):
            train_X[i, j] = datasets[0][i, j]
        train_Y[i, datasets[0][i, -1]] = 1
        
    print 'train_X.shape = {}'.format(train_X.shape)
    print 'train_Y.shape = {}'.format(train_Y.shape)


    # Validation data preparation
    Nv = datasets[1].shape[0]

    # For each word write a word index (not vector) to X tensor
    val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
    val_Y = np.zeros((Nv, 2), dtype=np.int)
    for i in xrange(Nv):
        for j in xrange(conv_input_height):
            val_X[i, j] = datasets[1][i, j]
        val_Y[i, datasets[1][i, -1]] = 1
        


    # Number of feature maps (outputs of convolutional layer)
    N_fm = 300
    # kernel size of convolutional layer
    kernel_size = 5 


    sampleSize          = datasets[0].shape[0]
    featureSize         = datasets[0].shape[1] 
    embeddingInputSize  = W.shape[0]
    embeddingOutputSize = W.shape[1]

    print 'sample           size: {}'.format(sampleSize           )
    print 'feature          size: {}'.format(featureSize          )
    print 'embedding input  size: {}'.format(embeddingInputSize   )
    print 'embedding output size: {}'.format(embeddingOutputSize  )


    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim    = W.shape[0], 
                        output_dim   = W.shape[1], 
                        input_length = conv_input_height,
                        weights      = [W], 
                        W_constraint = unitnorm()))
                        
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    #model.add(Reshape((conv_input_height, conv_input_width, 1)))
    model.add(Reshape((1, conv_input_height, conv_input_width)))

    # first convolutional layer
    model.add(Convolution2D(N_fm, 
                            kernel_size, 
                            conv_input_width, 
                            border_mode='valid', 
                            W_regularizer=l2(0.0001)))
                            


    # ReLU activation
    model.add(Activation('relu'))

    # aggregate data in every feature map to scalar using MAX operation
    model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1)))

    model.add(Flatten())
    model.add(Dropout(1))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(2))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    model.add(Activation('softmax'))

    # Custom optimizers could be used, though right now standard adadelta is employed
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', 
                optimizer=opt,
                metrics=['accuracy'])
                
    epoch = 0
    val_acc = []
    val_auc = []



    N_epoch = 1 

    for i in xrange(N_epoch):
        model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1)
        output = model.predict_proba(val_X, batch_size=10, verbose=1)
        # find validation accuracy using the best threshold value t
        vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
        # find validation AUC
        vauc = roc_auc_score(val_Y, output)
        val_acc.append(vacc)
        val_auc.append(vauc)
        print 'Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc)
        epoch += 1
        
    print '{} epochs passed'.format(epoch)
    print 'Accuracy on validation dataset:'
    print val_acc
    print 'AUC on validation dataset:'
    print val_auc


    # save model and weight
    # save model
    model_json = model.to_json()
    with open("model_cnn_sentiment.json", "w") as json_file:
        json_file.write(model_json)
    # save model weight
    model.save_weights('model_cnn_sentiment.h5')

    print("Saved model to disk")



def predict():
    '''
    predict on new sentences
    '''
    # load json and create model
    with open('model_cnn_sentiment.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_cnn_sentiment.h5")
    print("Loaded model from disk")

    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', 
                optimizer=opt,
                metrics=['accuracy'])


    
    x = cPickle.load(open("imdb-train-val-test.pickle", "rb"))
    revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    datasets = make_idx_data(revs, word_idx_map, max_l=2633, kernel_size=5)
    conv_input_height = int(datasets[0].shape[1]-1)
    Nv = datasets[1].shape[0]
    val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
    val_Y = np.zeros((Nv, 2), dtype=np.int)
    for i in xrange(Nv):
        for j in xrange(conv_input_height):
            val_X[i, j] = datasets[1][i, j]
        val_Y[i, datasets[1][i, -1]] = 1
        

    output = model.predict_proba(val_X, batch_size=10, verbose=1)
    print output



if __name__ == '__main__':
    #preprocessing()
    #learning()
    predict()



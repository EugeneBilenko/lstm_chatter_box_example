import tensorflow as tf
import numpy as np
from core.chatter_box_lstm import clean_text, loadStuff
from core.seq_to_seq import batch_size
from config import pickle_store

"""
- question:
 who are they
- neurojerk:
 i am an information gatherer i do not know prague i know that i would to as a fan

- question:
 What is it?
- neurojerk:
 i do not brain around physically here

"""

vocab_to_int = loadStuff("{}/{}".format(pickle_store, "vocab_to_int.p"))


def text_to_seq(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


def beta_predict():
    int_to_vocab = loadStuff("{}/{}".format(pickle_store, "int_to_vocab.p"))
    input_sentences=["who are they", "What is it?"]
    generagte_summary_length =  [3,2]

    texts = [text_to_seq(input_sentence) for input_sentence in input_sentences]
    checkpoint = "ddd/best_model.ckpt"
    if type(generagte_summary_length) is list:
        if len(input_sentences)!=len(generagte_summary_length):
            raise Exception("[Error] makeSummaries parameter generagte_summary_length must be same length as input_sentences or an integer")
        generagte_summary_length_list = generagte_summary_length
    else:
        generagte_summary_length_list = [generagte_summary_length] * len(texts)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        #Multiply by batch_size to match the model's input parameters
        for i, text in enumerate(texts):
            generagte_summary_length = generagte_summary_length_list[i]
            answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                              # summary_length: [generagte_summary_length],
                                              summary_length: [np.random.randint(15,28)],
                                              text_length: [len(text)]*batch_size,
                                              keep_prob: 1.0})[0]
            # Remove the padding from the summaries
            pad = vocab_to_int["<PAD>"]
            print('- question:\n\r {}'.format(input_sentences[i]))
            print('- neurojerk:\n\r {}\n\r\n\r'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))

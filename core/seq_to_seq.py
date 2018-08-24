import tensorflow as tf

from tensorflow.python.layers.core import Dense
from config import global_batch_size, max_line_length

def model_inputs():
    '''Create palceholders for inputs to the model'''
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    answer_length = tf.placeholder(tf.int32, (None,), name='answer_length')
    max_answer_length = tf.reduce_max(answer_length, name='max_dec_len')
    # Sequence length will be the max line length for each batch
    questions_length = tf.placeholder_with_default(max_line_length, None, name='questions_length')

    return input_data, targets, lr, keep_prob, answer_length, max_answer_length, questions_length




def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [global_batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([global_batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, questions_length, num_layers, rnn_inputs, keep_prob):
    # '''Create the encoding layer'''
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=keep_prob)
            print(questions_length)
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                    cell_bw=cell_bw,
                                                                    inputs=rnn_inputs,
                                                                    # todo predict error was here
                                                                    # sequence_length=questions_length,
                                                                    dtype=tf.float32)
            enc_output = tf.concat(enc_output, 2)
            rnn_inputs = enc_output

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, answer_length, dec_cell, output_layer,
                            vocab_size, max_answer_length,batch_size):
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=answer_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                       helper=training_helper,
                                                       initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=10),
                                                       output_layer = output_layer)

    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_answer_length)
    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, output_layer,
                             max_answer_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [global_batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        dec_cell.zero_state(dtype=tf.float32, batch_size=global_batch_size),
                                                        output_layer)

    inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                         output_time_major=False,
                                                         impute_finished=True,
                                                         maximum_iterations=max_answer_length)

    return inference_logits


def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)


def decoding_layer_v2(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, questions_length, answer_length,
                   max_answer_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    dec_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size, keep_prob) for _ in range(num_layers)])
    output_layer = Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     answer_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,answer_length,dec_cell,
                                                  output_layer,
                                                  vocab_size,
                                                  max_answer_length,
                                                  global_batch_size)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    output_layer,
                                                    max_answer_length,
                                                    global_batch_size)
    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, keep_prob, batch_size, questions_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int,answer_length, max_answer_length):
    '''Use the previous functions to create the training and inference logits'''

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       answers_vocab_size + 1,
                                                       enc_embedding_size,
                                                       initializer=tf.random_uniform_initializer(0, 1))
    enc_output, enc_state = encoding_layer(rnn_size, questions_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size + 1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits = decoding_layer_v2(dec_embed_input,
                                                       dec_embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       questions_vocab_size,
                                                       questions_length,
                                                       answer_length,
                                                       max_answer_length,
                                                       rnn_size,
                                                       questions_vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers)
    return training_logits, inference_logits

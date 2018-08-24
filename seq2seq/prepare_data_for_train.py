import time

from config import *
from seq2seq.helpers.pickle_actions import loadStuff
import numpy as np
import tensorflow as tf
from seq2seq.seq2seq_model import seq2seq_model, model_inputs


def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size, vocab_to_int):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch, vocab_to_int))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch, vocab_to_int))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


def start():
    sorted_answers = loadStuff("{}/{}".format(pickle_store, "sorted_answers.p"))
    sorted_questions = loadStuff("{}/{}".format(pickle_store, "sorted_questions.p"))
    word_embedding_matrix = loadStuff("{}/{}".format(pickle_store, "word_embedding_matrix.p"))

    vocab_to_int = loadStuff("{}/{}".format(pickle_store, "vocab_to_int.p"))
    int_to_vocab = loadStuff("{}/{}".format(pickle_store, "int_to_vocab.p"))

    # Set the Hyperparameters
    epochs = 30
    batch_size = 10
    rnn_size = 256
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.95

    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        # Load the model inputs
        input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

        # Create the training and inference logits
        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets,
                                                          keep_prob,
                                                          text_length,
                                                          summary_length,
                                                          max_summary_length,
                                                          len(vocab_to_int) + 1,
                                                          rnn_size,
                                                          num_layers,
                                                          vocab_to_int,
                                                          batch_size,
                                                          word_embedding_matrix)

        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits[0].rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')

        # Create the weights for sequence_loss, the sould be all True across since each batch is padded
        masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    print("Graph is built.")
    graph_location = "tb"
    print(graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(train_graph)

    # Subset the data for training
    start = 0
    end = len(sorted_answers)
    sorted_summaries_short = sorted_answers[start:end]
    sorted_texts_short = sorted_questions[start:end]
    print("The shortest text length:", len(sorted_texts_short[0]))
    print("The longest text length:", len(sorted_texts_short[-1]))
    # Train the Model
    learning_rate_decay = 0.95
    min_learning_rate = 0.0005
    display_step = 20  # Check training loss after every 20 batches
    stop_early = 0
    stop = 13  # If the update loss does not decrease in 3 consecutive update checks, stop training
    per_epoch = 3  # Make 3 update checks per epoch
    update_check = (len(sorted_texts_short) // batch_size // per_epoch) - 1

    update_loss = 0
    batch_loss = 0
    summary_update_loss = []  # Record the update losses for saving improvements in the model

    checkpoint = "ddd/best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        # If we want to continue training a previous session

        for epoch_i in range(1, epochs + 1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    get_batches(sorted_summaries_short, sorted_texts_short, batch_size, vocab_to_int)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: texts_batch,
                     targets: summaries_batch,
                     lr: learning_rate,
                     summary_length: summaries_lengths,
                     text_length: texts_lengths,
                     keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(sorted_texts_short) // batch_size,
                                  batch_loss / display_step,
                                  batch_time * display_step))
                    batch_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss / update_check, 3))
                    summary_update_loss.append(update_loss)

                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            if stop_early == stop:
                print("Stopping Training.")
                break

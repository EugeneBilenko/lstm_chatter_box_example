import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
from core.seq_to_seq import *
import pickle


def pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_line_length = 2
max_line_length = 25
threshold = 3


def initial_actions():
    with open('a', mode="r") as a:
        answers = a.readlines()

    with open('q', mode="r") as q:
        questions = q.readlines()
        # print(questions)

    # Clean the data
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    # print(clean_questions)

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # print(clean_answers)

    # Filter out the questions that are too short/long
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in clean_questions:
        if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    # Filter out the answers that are too short/long
    short_questions = []
    short_answers = []

    i = 0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_line_length and len(answer.split()) < max_line_length:
            short_answers.append(answer)
            short_questions.append(short_questions_temp[i])
        i += 1

    # Compare the number of lines we will use with the total number of lines.
    print("# of questions:", len(short_questions))
    print("# of answers:", len(short_answers))
    print("% of data used: {}%".format(round(len(short_questions) / len(questions), 4) * 100))

    # Create a dictionary for the frequency of the vocabulary
    vocab = {}
    for question in short_questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in short_answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Remove rare words from the vocabulary.
    # We will aim to replace fewer than 5% of words with <UNK>
    # You will see this ratio soon.
    count = 0
    for k, v in vocab.items():
        if v >= threshold:
            count += 1

    print("Size of total vocab:", len(vocab))
    print("Size of vocab we will use:", count)
    return vocab, short_answers, short_questions


def data_to_int():
    vocab, short_answers, short_questions = initial_actions()

    # In case we want to use a different vocabulary sizes for the source and target text,
    # we can set different threshold values.
    # Nonetheless, we will create dictionaries to provide a unique integer for each word.
    questions_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            questions_vocab_to_int[word] = word_num
            word_num += 1

    answers_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            answers_vocab_to_int[word] = word_num
            word_num += 1

    # Add the unique tokens to the vocabulary dictionaries.
    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

    for code in codes:
        questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1

    for code in codes:
        answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

    # Create dictionaries to map the unique integers to their respective words.
    # i.e. an inverse dictionary for vocab_to_int.
    questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
    answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

    # Check the length of the dictionaries.
    print(len(questions_vocab_to_int))
    print(len(questions_int_to_vocab))
    print(len(answers_vocab_to_int))
    print(len(answers_int_to_vocab))

    # Add the end of sentence token to the end of every answer.
    for i in range(len(short_answers)):
        short_answers[i] += ' <EOS>'

    # Convert the text to integers.
    # Replace any words that are not in the respective vocabulary with <UNK>
    questions_int = []
    for question in short_questions:
        ints = []
        for word in question.split():
            if word not in questions_vocab_to_int:
                ints.append(questions_vocab_to_int['<UNK>'])
            else:
                ints.append(questions_vocab_to_int[word])
        questions_int.append(ints)

    answers_int = []
    for answer in short_answers:
        ints = []
        for word in answer.split():
            # print(word)
            if word not in answers_vocab_to_int:
                ints.append(answers_vocab_to_int['<UNK>'])
            else:
                ints.append(answers_vocab_to_int[word])
        answers_int.append(ints)

    # Check the lengths
    print(len(questions_int))
    print(len(answers_int))

    pickleStuff("answers_int_to_vocab.p", answers_int_to_vocab)
    pickleStuff("questions_int_to_vocab.p", questions_int_to_vocab)

    # Calculate what percentage of all words have been replaced with <UNK>
    word_count = 0
    unk_count = 0

    for question in questions_int:
        for word in question:
            if word == questions_vocab_to_int["<UNK>"]:
                unk_count += 1
            word_count += 1

    for answer in answers_int:
        for word in answer:
            if word == answers_vocab_to_int["<UNK>"]:
                unk_count += 1
            word_count += 1

    unk_ratio = round(unk_count / word_count, 4) * 100

    print("Total number of words:", word_count)
    print("Number of times <UNK> is used:", unk_count)
    print("Percent of words that are <UNK>: {}%".format(round(unk_ratio, 3)))

    return questions_vocab_to_int, answers_vocab_to_int, questions_int, answers_int


def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    questions_vocab_to_int, answers_vocab_to_int, _, __ = data_to_int()

    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))

        # Need the lengths for the _lengths parameters summary = answer
        pad_answers_lengths = []
        for answer in pad_answers_batch:
            pad_answers_lengths.append(len(answer))
        
        pad_questions_lengths = []
        for question in pad_questions_batch:
            pad_questions_lengths.append(len(question))

        yield pad_questions_batch, pad_answers_batch, pad_answers_lengths, pad_questions_lengths


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"\n", "", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


def run_run():
    print('ebi swiney')
    if os.path.exists("sorted_answers.p"):
        print("LOAD FROM FILE")
        sorted_answers = loadStuff("sorted_answers.p")
        sorted_questions = loadStuff("sorted_questions.p")
        questions_vocab_to_int = loadStuff("questions_vocab_to_int.p")
        answers_vocab_to_int = loadStuff("answers_vocab_to_int.p")
    else:
        print("WTF!&")
        questions_vocab_to_int, answers_vocab_to_int, questions_int, answers_int = data_to_int()
        # questions_int,

        # Sort questions and answers by the length of questions.
        # This will reduce the amount of padding during training
        # Which should speed up training and help to reduce the loss

        sorted_questions = []
        sorted_answers = []

        for length in range(1, max_line_length + 1):
            for i in enumerate(questions_int):
                if len(i[1]) == length:
                    sorted_questions.append(questions_int[i[0]])
                    sorted_answers.append(answers_int[i[0]])

        print(len(sorted_questions))
        print(len(sorted_answers))
        print()
        for i in range(3):
            print(sorted_questions[i])
            print(sorted_answers[i])
            print()

        pickleStuff("questions_vocab_to_int.p", questions_vocab_to_int)
        pickleStuff("answers_vocab_to_int.p", answers_vocab_to_int)

        pickleStuff("sorted_answers.p", sorted_answers)
        pickleStuff("sorted_questions.p", sorted_questions)

    # Validate the training with 10% of the data
    train_valid_split = int(len(sorted_questions) * 0.05)

    # Split the questions and answers into training and validating data
    train_questions = sorted_questions[train_valid_split:]
    train_answers = sorted_answers[train_valid_split:]

    valid_questions = sorted_questions[:train_valid_split]
    valid_answers = sorted_answers[:train_valid_split]

    # Reset the graph to ensure that it is ready for training
    tf.reset_default_graph()
    # Start the session
    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():

        # Load the model inputs
        input_data, targets, lr, keep_prob, answer_length, max_answer_length, questions_length = model_inputs()

        # Find the shape of the input data for sequence_loss
        input_shape = tf.shape(input_data)

        # Create the training and inference logits
        train_logits, inference_logits = seq2seq_model(
            input_data=tf.reverse(input_data, [-1]),
            target_data=targets,
            keep_prob=keep_prob,
            batch_size=batch_size,
            questions_length=questions_length,
            answers_vocab_size=len(answers_vocab_to_int),
            questions_vocab_size=len(questions_vocab_to_int),
            enc_embedding_size=100,
            dec_embedding_size=100,
            rnn_size=rnn_size,
            num_layers=num_layers,
            questions_vocab_to_int=questions_vocab_to_int,
            answer_length=answer_length,
            max_answer_length=max_line_length
        )

        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(train_logits[0].rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')

        # Create the weights for sequence_loss, the sould be all True across since each batch is padded
        masks = tf.sequence_mask(answer_length, max_answer_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            # learning_rate = tf.Variable(tf.float32)
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    print("Graph is built.")
    graph_location = "tb/"
    print(graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(train_graph)

    # Train the Model
    learning_rate_decay = 0.95
    min_learning_rate = 0.0001
    display_step = 20  # Check training loss after every 20 batches
    stop_early = 0
    stop = 20  # If the update loss does not decrease in 20 consecutive update checks, stop training
    per_epoch = 3  # Make 3 update checks per epoch
    update_check = (len(sorted_questions) // batch_size // per_epoch) - 1

    update_loss = 0
    batch_loss = 0
    summary_update_loss = []  # Record the update losses for saving improvements in the model

    sess = tf.Session()
    checkpoint = "ddd/best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        learning_rate = 0.01

        for epoch_i in range(1, epochs + 1):
            for batch_i, (questions_batch, answers_batch, answers_lengths, questions_lengths) in enumerate(
                    batch_data(train_questions, train_answers, batch_size)):
                start_time = time.time()

                """
                print(questions_batch)
                # [[1605 4277 1361]]

                print(answers_batch)
                # [[1361 2095 5436]]

                print(answers_lengths)
                # [3]

                print(questions_lengths)
                # [9]
                """
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: questions_batch,
                     targets: answers_batch,
                     lr: learning_rate,

                     answer_length: answers_lengths,
                     questions_length: questions_lengths,

                     keep_prob: keep_probability,
                     })

                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_questions) // batch_size,
                                  update_loss / display_step,
                                  batch_time * display_step))
                    update_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    total_valid_loss = 0
                    start_time = time.time()
                    for batch_ii, (questions_batch, answers_batch, answers_lengths, questions_lengths) in \
                            enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                        valid_loss = sess.run(
                            cost, {
                                    input_data: questions_batch,
                                    targets: answers_batch,
                                    questions_length: questions_lengths,
                                    answer_length: answers_lengths,
                                   keep_prob: 1
                            })
                        total_valid_loss += valid_loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
                    print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

                    # Reduce learning rate, but not below its minimum value
                    learning_rate *= learning_rate_decay
                    if learning_rate < min_learning_rate:
                        learning_rate = min_learning_rate

                    summary_update_loss.append(avg_valid_loss)
                    if avg_valid_loss <= min(summary_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break

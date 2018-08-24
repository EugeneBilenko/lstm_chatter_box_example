SOURCE_DIR = "src"
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1

epochs = 50
batch_size = 10
global_batch_size = 10
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75
max_line_length = 25


# text lstm config
training_text_file = 'src/belling_the_cat.txt'
n_input = 3
hidden_size = 520  # same as state size


embedings_src = "src/numberbatch-en.txt"
questions_file = "q"
answers_file = "a"
pickle_store = "src/pickles_store"

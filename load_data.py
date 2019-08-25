import tensorflow as tf
import numpy as np

def split_input_target(chunk, vocab_size):
    '''
    Creates input & output label for training the data.
    chunk -- text to be processed for creating input output data
    input_text -- list of training data
    output_text -- list of output data
    '''
    input_text = chunk[:-1]
    output_text = chunk[1:]
    input_text = tf.one_hot(input_text, vocab_size)
    return input_text, output_text

def build_dataset(text_as_int,seq_length, vocab_size):
    '''
    Create a iteratable dataset from the raw text for training a model
    '''
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    char_dataset = char_dataset.batch(seq_length+1, drop_remainder=True)
    char_dataset = char_dataset.map(lambda x: split_input_target(x, vocab_size))
    return char_dataset

def load_txt (txt_path, show_text=False):
    '''
    Function to process as raw input text.
    txt_path - Path to input text
    show_txt - If true, displays first 500 characters from the input text
    '''
    text=open(txt_path,'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'. format(len(text)))
    vocab = sorted(set(text))
    vocab_size=len(vocab)
    print("{} unique characters".format(vocab_size))
    char2idx = {u:i for i,u in enumerate(vocab)}
    idx2char=np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    if show_text:
        print('\nFirst 500 characters from the text\n')
        print(text[:500])
    return text_as_int, vocab_size, char2idx, idx2char

def dataset(text_as_int, vocab_size, seq_length=50, batch_size=64):
    '''
    Shuffles the dataset and load the data in batch_size
    text_as_int -- Text to be trained mapped to integers
    vocab_size -- unique characters in the input text
    '''
    examples_per_epoch = len(text_as_int)//seq_length
    dataset = build_dataset(text_as_int, seq_length, vocab_size)
    dataset = dataset.shuffle(examples_per_epoch)
    dataset = dataset.batch(batch_size)
    return dataset
    
    
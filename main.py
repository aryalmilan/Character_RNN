import argparse
from load_data import *
from model import *
from generate_text import *


BATCH_SIZE=64
seq_length=100

def train_generate(path_txt):
    '''
    Function to train the model, and generate the text
    path_txt -- Path to the file to train the model
    '''
    text_as_int, vocab_size, char2idx, idx2char= load_txt (path_txt, show_text=False)
    data= dataset(text_as_int,vocab_size, seq_length=seq_length, batch_size=BATCH_SIZE)
    model=mymodel(vocab_size)
    examples_per_epoch = len(text_as_int)// seq_length
    steps_per_epoch = (examples_per_epoch//BATCH_SIZE)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(data.repeat(), steps_per_epoch = steps_per_epoch, epochs=5, verbose=2)
    start_text = input('Enter the start of your text : ')
    generate_txt(model,start_text,char2idx,idx2char,vocab_size,seq_length=seq_length, num_chars=1000)
    print('\n')
    
def generate(texttype):
    '''
    Generate the text on pretrained model. Models are pretrained on Shakespear sonnets and play
    texttype - Poem or Play
    '''
    start_text = input('Enter the start of your text : ')
    if texttype.lower()=='poem':
        path_txt='data/sonnets.txt'
        text_as_int, vocab_size, char2idx, idx2char= load_txt (path_txt, show_text=False)
        seq_length = 50
        model = tf.keras.models.load_model("models/shakespear_poem.h5")
    else:
        path_txt='data/shakespeare.txt'
        text_as_int, vocab_size, char2idx, idx2char= load_txt (path_txt, show_text=False)
        seq_length = 100
        model=tf.keras.models.load_model("models/shakespear_play.h5")
    generate_txt(model,start_text,char2idx,idx2char,vocab_size,seq_length=seq_length, num_chars=1000) 
    print('\n')
    
    
    
def main(mode, generate_type, file_path):
    if mode.lower()=='generate':
        generate(generate_type)
    else:
        train_generate(file_path)
        

if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Train and Generate text using char level RNN')
    ap.add_argument('mode',type = str, choices=['Train', 'Generate'], 
                    help='Train - Choose to train your own text and generate text\nGenerate-\
                    Use trained model to generate Shakespeare play or poem')
    ap.add_argument('--text_type', choices=['Poem','Play'], default='Poem',
                    help="Choose the type of text you want to generate. Play or Poem" )
    ap.add_argument('--file_path', type = str, help="Path to text file to be trained when in Train mode.")
    args = ap.parse_args()
    mode = args.mode
    file_path = None
    if args.text_type:
        generate_type = args.text_type
    if args.file_path:
        file_path = args.file_path
    if mode == 'Train' and file_path == None:
        ap.error('Missing training text file')
        
    main(mode, generate_type, file_path)
    
    
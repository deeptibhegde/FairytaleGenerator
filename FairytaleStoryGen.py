from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# load cleaned text sequences
in_filename = 'fairytaleSequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('modelF.h5')

# load the tokenizer
tokenizer = load(open('tokenizerF.pkl', 'rb'))
k = 1;
f = open("Your_Story.txt","a")
while(k):

	seed_text = input("\n\nEnter your sentence:")
	n = int(input("\nHow many words would you like to generate?"))

	# select a seed text
	#seed_text = "Mary went inside the room.  She was very tired and wanted to"
	#seed_text = lines[randint(0,len(lines))]
	print(seed_text + '\n')
	
	f.write(seed_text + " ")


	# generate new text
	generated = generate_seq(model, tokenizer, seq_length, seed_text, n)
	print(generated)
	f.write(generated)
#	f.write(".  ")
	
	again = input("Would you like to continue? y/n: ")
	if again == 'n':
		k = 0

f.close()

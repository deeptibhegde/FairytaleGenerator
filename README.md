# FairytaleGenerator
An interactive fairytale generator using LSTM networks

This is an LSTM network trained on public domain fairytales sourced from Project Gutenberg -  https://www.gutenberg.org/

The LSTM network is coded in keras and based on this tutorial - https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

Training the model requires tensorflow and keras. Use conda for an easy install - https://anaconda.org/conda-forge/keras

The training script is available, if you wish to retrain the model with your own data. Otherwise, the model file along with the sequences file enables you to run prediction inferences.

The script allows you to give a begin a story by providing half of a sentence. For example - "Johnny went to the market one day and saw". The generator will fill in the rest of the sentence. Both the prompt and generated words are logged into YourStory.txt to provide a nonsensical fairytale.

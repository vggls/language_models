Hyperparameter configuration of current LSTM models.

- embedding_dim = 300
- num_layers = 2 (the number of lstm layers)
- hidden_dim = 512
- dropout_rate = 0.3
- criterion = nn.CrossEntropyLoss()
- optimizer = Adam
- learning_rate = 0.001
- scheduler = None
- sequence_length = 50 (the number of words used to predict the next word)
- batch_size = 128 (we train the model in batches of 128 51-word sequences (sequence_length=50 words, target=1 word)
- epochs = 30 (the maximum number of epochs)
- patience = 10 (we monitor the validation perplexity and train as long as there is improvement within 10 epochs from the last improved value)

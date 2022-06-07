import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from CharRNN import RNN
import chars2vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    ############ Hyperparameters ############
    hidden_size = 512   # size of hidden state
    num_layers = 3      # num of layers in LSTM layer stack
    
    # load_path = "./preTrained/CharRNN_shakespeare.pth"
    # data_path = "./data/shakespeare.txt"
    # load_path = "./preTrained/CharRNN_sherlock.pth"
    # data_path = "./data/sherlock.txt"

    load_path = "./preTrained/CharRNN_rockyou.pth"
    data_path = "./data/flitered_rockyou.txt"
    #########################################
    
    # load the text file
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)-set('\n')))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
    
    # char to index and idex to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # convert data from chars to indices
    # data = list(data)
    # for i, ch in enumerate(data):
    #     data[i] = char_to_ix[ch]
    
    # data tensor on device
    # data = torch.tensor(data).to(device)
    # data = torch.unsqueeze(data, dim=1)
    
    # create and load model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)
    rnn.load_state_dict(torch.load(load_path))
    print("Model loaded successfully !!")
    
    # initialize variables

    # randomly select an initial string from the data
    # rand_index = np.random.randint(data_size - 11)
    # input_seq = data[rand_index : rand_index + 9]
    #
    # # compute last hidden state of the sequence
    # _, hidden_state = rnn(input_seq, hidden_state)
    #
    # # next element is the input to rnn
    # input_seq = data[rand_index + 9 : rand_index + 10]

    c2v_model = chars2vec.load_model('eng_50')

    num_honey_password = 10
    new_passwords = []
    for i in range(num_honey_password):
        data_ptr = 0
        hidden_state = None
        # TestData = "Alice0709"
        TestData = "Bobfacebook"
        # TestData = "Alice0709"
        keep_len = 5
        op_seq_len = 4   # total num of characters in output test sequence
        # print(TestData[:keep_len], end='')
        new_password = TestData[:keep_len]

        TestData = list(TestData)
        for i, ch in enumerate(TestData):
            TestData[i] = char_to_ix[ch]

        TestData = torch.tensor(TestData).to(device)
        TestData = torch.unsqueeze(TestData, dim=1)

        input_seq = TestData[0:keep_len]
        _, hidden_state = rnn(input_seq, hidden_state)
        input_seq = TestData[keep_len-1:keep_len]

        # generate remaining sequence
        # print("----------------------------------------")
        while True:
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample().item()
            
            # print the sampled character
            # print(ix_to_char[index], end='')
            new_password += ix_to_char[index]
            
            # next input is current output
            input_seq[0][0] = index
            data_ptr += 1
            
            if data_ptr >= op_seq_len:
                break

        print(new_password)
        new_passwords.append(new_password)
        print("----------------------------------------")
    
    
    # Create word embeddings
    word_embeddings = c2v_model.vectorize_words(new_passwords)
    print(word_embeddings.shape)

    base_word_emb = c2v_model.vectorize_words(TestData)
    distance = 


if __name__ == '__main__':
    test()


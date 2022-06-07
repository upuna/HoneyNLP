import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())
    
def train():
    ########### Hyperparameters ###########
    hidden_size = 512  # size of hidden state
    seq_len = 10       # length of LSTM sequence
    num_layers = 3      # num of layers in LSTM layer stack
    lr = 0.002          # learning rate
    epochs = 100        # max number of epochs
    op_seq_len = 10    # total num of characters in output test sequence
    load_chk = False    # load weights from save_path directory to continue training

    dzh = True
    # dzh = False

    if dzh:
        save_path = "./preTrained/CharRNN_rockyou.pth"
        data_path = "./data/flitered_rockyou.txt"
        # data_path = "./data/test.txt"
        data = open(data_path, 'r', encoding="latin-1").read()
        mydata = data.split("\n")
        chars = sorted(list(set(data)-set('\n')))
        word_size, data_size, vocab_size = len(mydata), len(data), len(chars)
        print("----------------------------------------")
        print("Data has {} passwords, {} characters, {} unique".format(word_size, data_size, vocab_size))
        print("----------------------------------------")

        # char to index and index to char maps
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }

        # passwords = []
        # # num = 1000
        # for i, word in enumerate(mydata):
        #     password = []
        #     for j, ch in enumerate(word):
        #         password.append(char_to_ix[ch])
        #     passwords.append(password)
        #     # if i>num:
        #     #     break
        # with open("passwords", "wb") as fp:   #Pickling
        #     pickle.dump(passwords, fp)

        with open("passwords", "rb") as fp:   #Pickling
            passwords = pickle.load(fp)

    else:
        save_path = "./preTrained/CharRNN_shakespeare.pth"
        data_path = "./data/shakespeare.txt"
        data = open(data_path, 'r').read()
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("----------------------------------------")
        print("Data has {} characters, {} unique".format(data_size, vocab_size))
        print("----------------------------------------")
    
        # char to index and index to char maps
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
        # convert data from chars to indices
        data = list(data)
        for i, ch in enumerate(data):
            data[i] = char_to_ix[ch]

        # data tensor on device
        data = torch.tensor(data).to(device)
        data = torch.unsqueeze(data, dim=1)
    
    # model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)
    
    # load checkpoint if True
    if load_chk:
        rnn.load_state_dict(torch.load(save_path))
        print("Model loaded successfully !!")
        print("----------------------------------------")
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    

    np.random.randint(low=0,high=1000,size=10)

    # training loop
    if dzh:
        i_batch = 0
        batch_size = 10000
        while True:
            batch = np.random.randint(low=0,high=len(passwords),size=batch_size)
            n=0
            running_loss = 0
            last_loss = 0
            for i in range(batch_size):
                hidden_state = None
                password = torch.tensor(passwords[batch[i]]).to(device)
                input_seq = password[:-1].reshape(-1,1)
                target_seq = password[1:].reshape(-1,1)
                output, hidden_state = rnn(input_seq, hidden_state)
                loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
                running_loss += loss.item()
                n += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print loss and save weights after every epoch
            print("Batch: {0} \t Loss: {1:.8f}   ".format(i_batch, running_loss/n), end="")
            torch.save(rnn.state_dict(), save_path)
            if np.abs(running_loss/n - last_loss)<1e-4:
                break
            last_loss = running_loss/n
            i_batch += 1

            # Test
            number_of_honey_password = 10
            for j in range(number_of_honey_password):
                TestData = "Alice0709"
                keep_len = 5
                op_seq_len = 4
                data_ptr = 0
                print(TestData[:keep_len], end='')
                TestData = list(TestData)
                for i, ch in enumerate(TestData):
                    TestData[i] = char_to_ix[ch]

                TestData = torch.tensor(TestData).to(device)
                TestData = torch.unsqueeze(TestData, dim=1)

                input_seq = TestData[0:keep_len]
                _, hidden_state = rnn(input_seq, hidden_state)
                input_seq = TestData[keep_len:keep_len+1]

                while True:
                    # forward pass
                    output, hidden_state = rnn(input_seq, hidden_state)
                    
                    # construct categorical distribution and sample a character
                    output = F.softmax(torch.squeeze(output), dim=0)
                    dist = Categorical(output)
                    index = dist.sample().item()
                    
                    # print the sampled character
                    print(ix_to_char[index], end='')
                    
                    # next input is current output
                    input_seq[0][0] = index
                    data_ptr += 1
                    
                    if data_ptr  >= op_seq_len:
                        break
                print(" ", end="")
            print(" ")


    else:
        for i_epoch in range(1, epochs+1):
            # random starting point (1st 100 chars) from data to begin
            data_ptr = np.random.randint(100)
            n = 0
            running_loss = 0
            hidden_state = None
            while True:
                input_seq = data[data_ptr : data_ptr+seq_len]
                target_seq = data[data_ptr+1 : data_ptr+seq_len+1]
                
                # forward pass
                output, hidden_state = rnn(input_seq, hidden_state)
                
                # compute loss
                loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
                running_loss += loss.item()
                
                # compute gradients and take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # update the data pointer
                data_ptr += seq_len
                n +=1
                
                # if at end of data : break
                if data_ptr + seq_len + 1 > data_size:
                    break

                # print loss and save weights after every epoch
                print("Epoch: {0} \t Loss: {1:.8f}   ".format(i_epoch, running_loss/n), end="")
                torch.save(rnn.state_dict(), save_path)
                
                # sample / generate a text sequence after every epoch
                data_ptr = 0
                hidden_state = None
                
                # random character from data to begin
                rand_index = np.random.randint(data_size-1)
                input_seq = data[rand_index : rand_index+1]
                
                print("----------------------------------------")
                while True:
                    # forward pass
                    output, hidden_state = rnn(input_seq, hidden_state)
                    
                    # construct categorical distribution and sample a character
                    output = F.softmax(torch.squeeze(output), dim=0)
                    dist = Categorical(output)
                    index = dist.sample()
                    
                    # print the sampled character
                    print(ix_to_char[index.item()], end='')
                    
                    # next input is current output
                    input_seq[0][0] = index.item()
                    data_ptr += 1
                    
                    if data_ptr > op_seq_len:
                        break
                    
                print("\n----------------------------------------")
        
if __name__ == '__main__':
    train()



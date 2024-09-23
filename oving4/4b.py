import torch
import numpy as np
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size, emojis_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emojis_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


def encode(length):
    return [[1. if i == j else 0. for i in range(length)] for j in range(length)]


index_to_char = [' ', 'a', 't', 'c', 'h',
                 'o', 'f', 'l', 'm', 'n', 'p', 'r', 's']

char_encodings = encode(len(index_to_char))
print(char_encodings)
encoding_size = len(char_encodings)

emoji_dict = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F400',
    'cat': '\U0001F408',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

index_to_emoji = [emoji for emoji in emoji_dict.values()]
emoji_encodings = encode(len(index_to_emoji))
print(index_to_emoji)
emoji_encoding_size = len(emoji_encodings)

x_train = torch.tensor([
    [[char_encodings[4]], [char_encodings[1]], [
        char_encodings[2]], [char_encodings[0]]],  # 'hat '
    [[char_encodings[11]], [char_encodings[1]], [
        char_encodings[2]], [char_encodings[0]]],  # 'rat '
    [[char_encodings[3]], [char_encodings[1]], [
        char_encodings[2]], [char_encodings[0]]],  # 'cat '
    [[char_encodings[6]], [char_encodings[3]], [
        char_encodings[1]], [char_encodings[2]]],  # 'flat'
    [[char_encodings[8]], [char_encodings[3]], [
        char_encodings[2]], [char_encodings[2]]],  # 'matt'
    [[char_encodings[3]], [char_encodings[1]], [
        char_encodings[10]], [char_encodings[0]]],  # 'cap '
    [[char_encodings[12]], [char_encodings[5]], [
        char_encodings[9]], [char_encodings[0]]]   # 'son '
])


y_train = torch.tensor([
    [emoji_encodings[0] for i in range(4)],
    [emoji_encodings[1] for i in range(4)],
    [emoji_encodings[2] for i in range(4)],
    [emoji_encodings[3] for i in range(4)],
    [emoji_encodings[4] for i in range(4)],
    [emoji_encodings[5] for i in range(4)],
    [emoji_encodings[6] for i in range(4)]])

print(x_train.shape)
print(x_train.shape)
model = LongShortTermMemoryModel(encoding_size, emoji_encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def generate_emoji(string):
    y = -1
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor(
            [[char_encodings[char_index]]], dtype=torch.float))
    print(y)
    print(string, "-> ", index_to_emoji[y.argmax(1)])


generate_emoji("rt")
generate_emoji("rats")
generate_emoji("h")
generate_emoji("htt")
generate_emoji("so")
generate_emoji("mat")
generate_emoji("ca")
generate_emoji("cat")
generate_emoji("capp")
generate_emoji("cats")

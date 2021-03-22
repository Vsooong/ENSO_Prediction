from torch import nn
import torch
from lib.util import make_layers, nino_index
from collections import OrderedDict
from lib.land_sea import land_mask

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=26):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).to(device)
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).to(device)
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [5, 64, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(12, 36), input_channels=64, filter_size=3, num_features=64),
        CLSTM_cell(shape=(6, 18), input_channels=64, filter_size=3, num_features=64),
    ]
]


# convlstm_decoder_params = [
#     [
#         OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
#         OrderedDict({
#             'conv2_leaky_1': [64, 16, 3, 1, 1],
#             'conv3_leaky_1': [16, 1, 1, 1, 0]
#         }),
#     ],
#
#     [
#         CLSTM_cell(shape=(12, 36), input_channels=64, filter_size=3, num_features=64),
#         CLSTM_cell(shape=(24, 72), input_channels=64, filter_size=3, num_features=64),
#     ]
# ]


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None, seq_len=seq_number)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return inputs


class convLSTM(nn.Module):
    def __init__(self, encoder=Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])):
        super().__init__()
        self.encoder = encoder
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1))
        self.rnn = nn.GRU(108, 24, batch_first=True)
        self.fc = nn.Linear(108 * 12, 24)

    def forward(self, sst, t300, ua, va):
        batch, month, h, w = sst.size()
        sea_mask = torch.as_tensor(~land_mask(), dtype=torch.float).repeat(batch, month, 1, 1).to(device)
        inputs = torch.stack([sst, t300, ua, va, sea_mask], dim=2)
        output = self.encoder(inputs)
        output = output.permute(1, 2, 0, 3, 4)
        output = self.conv1(output).squeeze(1)

        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output
        # output = torch.flatten(output, start_dim=2)
        # out, nino_index = self.rnn(output)
        # return nino_index.squeeze(0)


# class Decoder(nn.Module):
#     def __init__(self, subnets, rnns):
#         super().__init__()
#         assert len(subnets) == len(rnns)
#
#         self.blocks = len(subnets)
#
#         for index, (params, rnn) in enumerate(zip(subnets, rnns)):
#             setattr(self, 'rnn' + str(self.blocks - index), rnn)
#             setattr(self, 'stage' + str(self.blocks - index),
#                     make_layers(params))
#
#     def forward_by_stage(self, inputs, state, subnet, rnn):
#         inputs, state_stage = rnn(inputs, state, seq_len=26)
#         seq_number, batch_size, input_channel, height, width = inputs.size()
#         inputs = torch.reshape(inputs, (-1, input_channel, height, width))
#         inputs = subnet(inputs)
#         inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
#                                         inputs.size(2), inputs.size(3)))
#         return inputs
#
#         # input: 5D S*B*C*H*W
#
#     def forward(self, hidden_states):
#         inputs = self.forward_by_stage(None, hidden_states[-1],
#                                        getattr(self, 'stage2'),
#                                        getattr(self, 'rnn2'))
#         for i in list(range(1, self.blocks))[::-1]:
#             inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
#                                            getattr(self, 'stage' + str(i)),
#                                            getattr(self, 'rnn' + str(i)))
#         inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
#         return inputs


# class convLSTM(nn.Module):
#
#     def __init__(self, encoder=Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]),
#                  decoder=Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1])):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.nino_linear = nn.Linear(10, 24)
#
#     def forward(self, sst, t300, ua, va):
#         inputs = torch.stack([sst, t300, ua, va], dim=2)
#         state = self.encoder(inputs)
#         output = self.decoder(state)
#         nino_indexes = nino_index(output.squeeze(2))
#         # new_sst = torch.cat([sst, output.squeeze(2)[:, 0:2, ...]], dim=1)
#         sim_nino_index = nino_index(sst)
#         sim_nino_index = self.nino_linear(sim_nino_index)
#         nino_indexes = (sim_nino_index + nino_indexes) / 2
#         print(output.shape)
#         return output[:, :24].squeeze(2), nino_indexes


if __name__ == '__main__':
    devcie = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = convLSTM().to(devcie)
    input1 = torch.rand(4, 12, 24, 72).to(devcie)
    input2 = torch.rand(4, 12, 24, 72).to(devcie)
    input3 = torch.rand(4, 12, 24, 72).to(devcie)
    input4 = torch.rand(4, 12, 24, 72).to(devcie)
    ninos = model(input1, input2, input3, input4)

    print(ninos.shape)
    nParams = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print('number of parameters: %d' % nParams)

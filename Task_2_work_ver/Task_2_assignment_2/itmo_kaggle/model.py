from torch.utils.data import Dataset
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F


def prepare_shape(feature):
    tmp = feature
    N = 96 # размер во времени (можно увеличить, должно стать лучше)
    while tmp.shape[1] < N:
        # можно попробовать сделать np.pad для коротких файлов, вместо повторения до необходимой длины
        tmp = np.hstack((tmp, tmp))
    # случайный сдвиг должен улучшить результат (для этого нужно функцию перенести в EventDetectionDataset)
    length = tmp.shape[1]
    start = random.randint(0,length-N)
    tmp = tmp[np.newaxis, :, start:start+N]
    return tmp


class EventDetectionDataset(Dataset):
    def __init__(self, x, y=None, names = None):
        self.x = x
        self.y = y
        self.names= names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return prepare_shape(self.x[idx]), self.y[idx]
        return prepare_shape(self.x[idx]), self.names[idx]


class torch_model(nn.Module):
    def __init__(self, config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8)):
        super(torch_model, self).__init__()
        self.config = config
        self.fc1 = nn.Conv1d(in_channels= 1, out_channels=8, kernel_size=k_size[0])      #(batch_size, in_channels, seq_len)
        self.bn_1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(kernel_size=p_size[0])

        self.fc2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=k_size[1])
        self.bn_2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(kernel_size=p_size[1])

        self.fc3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=k_size[2])
        self.bn_3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool1d(kernel_size=p_size[2])

        self.fc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k_size[3])
        self.bn_4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool1d(kernel_size=p_size[3])                           #(batch_size, in_channels, seq_len)


        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.gap = nn.AdaptiveAvgPool1d(output_size = 1)
        self.dropout = nn.Dropout(p=0.9)
        self.linear = nn.Linear(128, self.config.n_classes)
        self.bn_final = nn.BatchNorm1d(self.config.n_classes)


    def forward(self, x, hidden):
        out = self.fc1(x)
        out = self.bn_1(out)
        out = self.relu1(out)
        out = self.mp1(out)

        out = self.fc2(out)
        out = self.bn_2(out)
        out = self.relu2(out)
        out = self.mp2(out)

        out = self.fc3(out)
        out = self.bn_3(out)
        out = self.relu3(out)
        out = self.mp3(out)

        out = self.fc4(out)
        out = self.bn_4(out)
        out = self.relu4(out)
        out = self.mp4(out)
        out = out.permute(0,2,1)

        out, hidden = self.lstm1(out, hidden)
        out, hidden = self.lstm2(out, hidden)

        out = out.permute(0,2,1)
        out = self.gap(out).squeeze()
        out = self.dropout(out)
        out = self.linear(out)
        out = self.bn_final(out)
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        # weight = next(self.parameters()).data
        hidden = (torch.zeros((1, batch_size, 128)).to('cuda'),
                      torch.zeros((1, batch_size, 128)).zero_().to('cuda'))

        return hidden


class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=(1, 1))

        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=96)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(137280, 41)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.dropout3(x)

        x = self.flat(x)
        # здесь можно еще добавить полносвязных слой или слои
        x = self.fc1(x)
        return x


# https://github.com/lukemelas/EfficientNet-PyTorch
class BaseLineModel(nn.Module):

    def __init__(self, sample_rate=16000, n_classes=41):
        super().__init__()
        # self.ms = torchaudio.transforms.MelSpectrogram(sample_rate)
        #         self.bn1 = nn.BatchNorm2d(1)

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3, padding=1)

        self.features = EfficientNet.from_pretrained('efficientnet-b0')
        # use it as features
        #         for param in self.features.parameters():
        #             param.requires_grad = False

        self.lin1 = nn.Linear(1000, 333)

        self.lin2 = nn.Linear(333, 111)

        self.lin3 = nn.Linear(111, n_classes)

    def forward(self, x):
        x = self.ms(x)
        #         x = self.bn1(x)

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn3(x))

        x = self.features(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(x)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x)
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # ?

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def _change_in_channels(model, in_channels):
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
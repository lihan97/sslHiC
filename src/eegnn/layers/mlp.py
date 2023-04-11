from torch import nn

class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        input_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        output_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        batch_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        input_dim: int
            Input dimension of the linear layer
        output_dim: int
            Output dimension of the linear layer
    """

    def __init__(self, input_dim, output_dim, activation, dropout=0., batch_norm=False, bias=True, bn_affine=True):
        super(FCLayer, self).__init__()

        # self.__params = locals()
        # del self.__params['__class__']
        # del self.__params['self']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim, affine=bn_affine)
        self.activation = activation
        self.bn_affine = bn_affine
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d) and self.bn_affine:
                nn.init.normal_(m.weight.data, mean=1, std=0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.output_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCnum_layers
    """

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, mid_activation, last_activation=None,
                 dropout=0., mid_batch_norm=False, last_batch_norm=False, bn_affine=True):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fully_connected = nn.ModuleList()
        if num_layers <= 1:
            self.fully_connected.append(FCLayer(input_dim, output_dim, activation=last_activation, batch_norm=last_batch_norm,
                                                dropout=dropout, bn_affine=bn_affine))
        else:
            self.fully_connected.append(FCLayer(input_dim, hidden_dim, activation=mid_activation, batch_norm=mid_batch_norm,
                                                dropout=dropout, bn_affine=bn_affine))
            for _ in range(num_layers - 2):
                self.fully_connected.append(FCLayer(hidden_dim, hidden_dim, activation=mid_activation,
                                                    batch_norm=mid_batch_norm, dropout=dropout, bn_affine=bn_affine))
            self.fully_connected.append(FCLayer(hidden_dim, output_dim, activation=last_activation, batch_norm=last_batch_norm,
                                                dropout=dropout, bn_affine=bn_affine))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
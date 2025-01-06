import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    """
    Network to generate the weights of the target network
    """

    def __init__(self, input_dim, hidden_dim, output_dim, init_scale, num_layers, use_layer_norm=True):
        """
        Initialize network

        args:
            input_dim (int): input layer dimension
            hidden_dim (int): hidden layer dimension
            output_dim (int): output layer dimension
            init_scale (float): scale for weight initialization
            num_layers (int): number of layers
            use_layer_norm (bool): indicate whether to use layer normalization
        """

        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.init_scale = init_scale

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Orthogonally initialize hypernetwork weights
        """

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=self.init_scale)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        Compute weights for target network

        args:
            x (torch.Tensor): network input [B, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x

class HyperRNNAgent(nn.Module):
    """
    Agent with HyperNetwork action decoder
    """

    def __init__(self, input_dim, args):
        """
        Initialize HyperRNNAgent

        args:
            input_dim (int): input dimension
            args.action_dim (int): output dimension
            args.hidden_dim (int): dimension of hidden layers
            args.dim_capabilities (int): dimension of capabilities, used for conditioning the hyper network
            args.hypernetwork_kwargs (dict): hypernetwork specific args
        """
        super().__init__()
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.dim_capabilities = args.dim_capabilities
        self.hypernet_kwargs = args.hypernet_kwargs

        # 1 layer encoder mlp
        self.encoder = nn.Linear(input_dim - self.dim_capabilities, self.hidden_dim)
        
        # GRU cell
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # weight and bias hypernetwork
        self.weight_hypernet = HyperNetwork(
            input_dim=self.hypernet_kwargs["INPUT_DIM"],
            hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"],
            output_dim=self.hidden_dim * self.action_dim,
            init_scale=self.hypernet_kwargs["INIT_SCALE"],
            num_layers=self.hypernet_kwargs["NUM_LAYERS"],
            use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"]
        )
        self.bias_hypernet = HyperNetwork(
            input_dim=self.hypernet_kwargs["INPUT_DIM"],
            hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"],
            output_dim=self.action_dim,
            init_scale=0.0,
            num_layers=self.hypernet_kwargs["NUM_LAYERS"],
            use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"]
        )
    
    def init_hidden(self):
        """
        Init hidden states
        """

        return self.encoder.weight.new(1, self.hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        """
        Get agent Q values

        args:
            obs: agent observation (batch_size, input_dim)
            hidden_state: agent gru hidden state (batch_size, hidden_dim)

        returns:
            q_values: action q values (batch_size, action_dim)
            hidden_state: updated hidden state (batch_size, hidden_dim)
        """

        batch_size = obs.size(0)
        capabilities = obs[:, -self.dim_capabilities:]
        observations = obs[:, :-self.dim_capabilities]

        # encode observations
        embedding = F.relu(self.encoder(observations))

        # update RNN hidden state
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        hidden_state = self.rnn(embedding, h_in)

        # generate weights and biases using hypernetworks (include obs + capabilities)
        weights = self.weight_hypernet(obs).view(batch_size, self.hidden_dim, self.action_dim)
        biases = self.bias_hypernet(obs).view(batch_size, 1, self.action_dim)

        # q = embedding @ weights + biases
        q_values = torch.bmm(hidden_state.unsqueeze(1), weights) + biases
        q_values = q_values.squeeze(1)  # Remove the extra dimension

        return q_values, hidden_state

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import numpy as np 
torch, nn = try_import_torch()

class PolicyLSTM(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=1024,
        lstm_state_size=512,
        num_layers = 2
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = obs_space.original_space["embedding"].shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size
        self.num_layers = num_layers

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, num_layers=num_layers ,batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            torch.tensor(np.zeros(self.lstm_state_size, np.float32)).squeeze(),
            torch.tensor(np.zeros(self.lstm_state_size, np.float32)).squeeze(),
        ] 
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ) :
        """Adds time dimension to batch before sending inputs to forward_rnn().
        
        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs"]["embedding"].float()
        # flat_inputs = obs.reshape(obs.shape[0], -1) 
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        logits = torch.reshape(output, [-1, self.num_outputs])
        logits = logits - (1_000_000 * input_dict["obs"]["actions_mask"])
        return logits, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]



class PolicyNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name,dropout_rate,policy_hidden_layers,vf_hidden_layers):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
    
        self.share_weights = model_config["vf_share_layers"]

        dropout = dropout_rate
  
        
        input_size = obs_space.original_space["embedding"].shape[0]

        # Policy network
        policy_hidden_sizes = policy_hidden_layers
        policy_output_size = num_outputs

        self.policy_layers = nn.ModuleList()
        policy_layers_sizes = [input_size] + policy_hidden_sizes
        for i in range(len(policy_layers_sizes) - 1):
            layer = nn.Linear(policy_layers_sizes[i],
                              policy_layers_sizes[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            self.policy_layers.append(layer)
            self.policy_layers.append(
                nn.BatchNorm1d(policy_layers_sizes[i + 1]))
            self.policy_layers.append(nn.Dropout(dropout))

        # Policy head
        self.logits_layer = nn.Linear(policy_layers_sizes[-1],
                                      policy_output_size)
        nn.init.xavier_uniform_(self.logits_layer.weight)

        # Value separate network
        value_hidden_sizes = vf_hidden_layers
        value_output_size = 1

        self.value_layers = nn.ModuleList()
        value_layers_sizes = [input_size] + value_hidden_sizes
        for i in range(len(value_layers_sizes) - 1):
            layer = nn.Linear(value_layers_sizes[i], value_layers_sizes[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            self.value_layers.append(layer)
            self.value_layers.append(nn.BatchNorm1d(value_layers_sizes[i + 1]))
            self.value_layers.append(nn.Dropout(dropout))

        # Value head
        if (self.share_weights):
            # If value function and policy weights are shared, use last weights of the policy network
            self.value_layer = nn.Linear(policy_layers_sizes[-1], value_output_size)
        else : 
            self.value_layer = nn.Linear(value_layers_sizes[-1], value_output_size)
        nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["embedding"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._last_flat_in
        for layer in self.policy_layers:
            if isinstance(layer, nn.Linear):
                self._last_flat_in = layer(self._last_flat_in)
            elif isinstance(layer, nn.BatchNorm1d):
                self._last_flat_in = layer(self._last_flat_in)
                self._last_flat_in = nn.functional.relu(self._last_flat_in)
            elif isinstance(layer, nn.Dropout):
                self._last_flat_in = layer(self._last_flat_in)
        # Output logits
        logits = self.logits_layer(self._last_flat_in)
        # Masking selected and restricted actions
        logits = logits - (1_000_000 * input_dict["obs"]["actions_mask"])
        return logits, state

    def value_function(self):
        if (self.share_weights):
            assert self._last_flat_in is not None, "must call forward() first"
            value = self.value_layer(self._last_flat_in)
        else :
            assert self._features is not None, "must call forward() first"
            self._value_features = self._features
            for layer in self.value_layers:
                if isinstance(layer, nn.Linear):
                    self._value_features = layer(self._value_features)
                elif isinstance(layer, nn.BatchNorm1d):
                    self._value_features = layer(self._value_features)
                    self._value_features = nn.functional.relu(self._value_features)
                elif isinstance(layer, nn.Dropout):
                    self._value_features = layer(self._value_features)
            # Output the value
            value = self.value_layer(self._value_features)
        return value.squeeze(1)
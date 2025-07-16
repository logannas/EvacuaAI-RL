import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, obs_size: int, n_actions: int) -> None:
        """
        Initializes the neural network with six fully connected layers,
        layer normalization, and dropout for improved learning stability.

        Args:
            obs_size (int): Number of input features.
            n_actions (int): Number of possible actions (output of the network).
        """
        super(Network, self).__init__()

        # First hidden layer: transforms from obs_size to 512 neurons
        self.fc1 = nn.Linear(obs_size, 512)
        self.ln1 = nn.LayerNorm(512)  # Layer normalization
        self.dropout1 = nn.Dropout(p=0.2)  # Dropout for regularization

        # Second hidden layer: reduces from 512 to 256 neurons
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(p=0.2)

        # Third hidden layer: reduces from 256 to 128 neurons
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(p=0.2)

        # Fourth hidden layer: reduces from 128 to 64 neurons
        self.fc4 = nn.Linear(128, 64)
        self.ln4 = nn.LayerNorm(64)
        self.dropout4 = nn.Dropout(p=0.1)

        # Fifth hidden layer: reduces from 64 to 32 neurons
        self.fc5 = nn.Linear(64, 32)

        # Output layer: produces a vector of size equal to the number of actions
        self.fc6 = nn.Linear(32, n_actions)

        # Set the device to GPU if available, otherwise use CPU
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using He initialization (Kaiming uniform).
        """
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]:
            nn.init.kaiming_uniform_(
                layer.weight, nonlinearity="leaky_relu"
            )  # Changed to leaky_relu
            nn.init.zeros_(layer.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Passes the input tensor through the neural network and returns action scores.

        Args:
            x (T.Tensor): Input tensor representing the state.

        Returns:
            T.Tensor: A vector with Q-values for each possible action.
        """
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        x = F.leaky_relu(self.ln4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)
        x = self.fc6(x)  # Output layer without activation (Q-values)

        return x

    def act(self, state: T.Tensor) -> int:
        """
        Selects the best action for a given state.

        Args:
            state (T.Tensor): Tensor representing the current state.

        Returns:
            int: Index of the selected action.
        """
        state = state.to(self.device)  # Ensure the state is on the correct device
        actions = self.forward(state)  # Get Q-values for each action
        action = T.argmax(actions).item()  # Select the action with the highest Q-value

        return action

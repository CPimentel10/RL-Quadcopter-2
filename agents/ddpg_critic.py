from keras import layers, models, optimizers
from keras import backend as K


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize a Critic instance."""
        self.state_size = state_size
        self.action_size = action_size
        # build the model
        self.build_model()

    def build_model(self):
        """Build a Critic (Value) model that maps (state, action)> Q_values."""
        # Input layers
        states = layers.Input(shape=(self.state_size,), name="states")
        actions = layers.Input(shape=(self.action_size,), name="actions")

        # Add some hidden layers to state pathway
        net_states = layers.Dense(units=32, activation="relu")(states)
        net_states = layers.Dense(units=64, activation="relu")(net_states)

        # Add some hidden layers to action pathway
        net_actions = layers.Dense(units=32, activation="relu")(actions)
        net_actions = layers.Dense(units=64, activation="relu")(net_actions)

        # Combine both pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        Q_values = layers.Dense(units=1, name='q_values')(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used
        # by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.inputs, K.learning_phase()],
            outputs=action_gradients)

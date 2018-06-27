from keras.models import Sequential
from keras.layers import Dense, Lambda, Input
from keras import backend as K
from keras import optimizers


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_high, action_low):
        """Initialize Actor instance."""
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = self.action_high - self.action_low

        self.build_model()

    def build_model(self):
        """Build the actor(policy model that maps states > actions)."""
        model = Sequential()
        model.add(Dense(input_dim=self.state_size, name="states"))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        # Add final output layer with sigmoid activation
        model.add(Dense(self.action_size,
                        activation="sigmoid", name="raw_actions"))
        # Scale [0, 1] output for each action dimension to proper range
        model.add(Lambda(lambda x: (x * self.action_range) +
                         self.action_low), name="actions")

        self.model = model

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * model.get_layer("actions"))

        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss)
        # Custom training function for our model.
        self.train_fn = K.function(inputs=[
                                   self.model.input, action_gradients,
                                   K.learning_phase()], outputs=[],
                                   updates=updates_op)

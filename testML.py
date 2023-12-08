import tensorflow as tf
import numpy as np
import copy

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lam=0.95, clip_epsilon=0.2, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.model_old = self._build_model()
        self.model_old.set_weights(self.model.get_weights())

    def _build_model(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        advantage = tf.keras.layers.Input(shape=(1,))
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,))

        x = tf.keras.layers.Dense(64, activation='relu')(state_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        out_actions = tf.keras.layers.Dense(self.action_size, activation='softmax', name='output')(x)

        model = tf.keras.models.Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=[self._ppo_loss(advantage, old_prediction)])
        return model

    def _ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -tf.keras.backend.mean(tf.keras.backend.minimum(r * advantage, tf.keras.backend.clip(r, min_value=1 - self.clip_epsilon, max_value=1 + self.clip_epsilon) * advantage) + 0.01 * (-prob * tf.keras.backend.log(prob + 1e-10)))
        return loss

    def update_old_model(self):
        self.model_old.set_weights(self.model.get_weights())

    def get_action(self, state):
        policy = self.model.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])[0]
        return np.random.choice(self.action_size, p=policy)

    def train(self, states, actions, advantages, old_predictions):
        self.model.fit([states, advantages, old_predictions], [actions], verbose=0)

    def get_gaes(self, rewards, dones, v_values, next_v_value):
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, v_values, next_v_value)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.lam * self.gamma * (1 - dones[t]) * gaes[t + 1]
        return gaes

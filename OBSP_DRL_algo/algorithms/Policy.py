
import numpy as np
import tensorflow as tf
import rusher.settings as settings
import random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Reduce warning messages

class Policy(object):
    """
     NN-based policy approximation
      """
    def __init__(self, obs_dim, act_dim, learningParameters, networkParameters):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
        """
        self.lParameters = learningParameters
        self.nParameters = networkParameters
        self.beta = 1.0  
        self.eta = 0
        self.lr_multiplier = 1.0  
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """
        Build and initialize TensorFlow graph
        """
        self.g = tf.Graph()
        with self.g.as_default() as gg:
            self._placeholders()
            if self.nParameters['network_type'] == 'convolutional':
                self._policy_conv_nn()
            else:
                self._policy_nn()
            self._logprob()
            self.scaleVariance()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """
        TF Placeholders
        """
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,self.obs_dim), name='obs')
        self.reshaped = tf.reshape(self.obs_ph, [-1, 64, 64, 1])
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_conv_nn(self):
        """
        Convolutional Neural network
        """
        self.lr = self.lParameters['lr_actor']

        out = tf.layers.conv2d(inputs=self.reshaped,
                                strides=self.nParameters['strides'][0],
                                filters=self.nParameters['filters'][0],
                                kernel_size=self.nParameters['kernel'][0],
                                activation=getattr(tf.nn, self.nParameters['activation_h_a']))

        for i, layer in enumerate(self.nParameters['filters'][1:]):
            print(i, layer)
            out = tf.layers.conv2d(inputs=out,
                                   strides=self.nParameters['strides'][i+1],
                                   filters=layer,
                                   kernel_size=self.nParameters['kernel'][i+1],
                                   activation=getattr(tf.nn, self.nParameters['activation_h_a']))  
        out = tf.layers.flatten(out)

        print('Flattened policy: {}, {}'.format(tf.shape(out), tf.size(out)))
        

        # Fully connected layers

        for layer in self.nParameters['hidden_layers_a']:
            out = tf.layers.dense(out, layer, activation=getattr(tf.nn, self.nParameters['activation_h_a']))


        # out = tf.layers.dense(out, 16, activation=tf.nn.relu)
        self.means = tf.layers.dense(out, self.act_dim, activation=getattr(tf.nn, self.nParameters['activation_o_a']),
                                     kernel_initializer=tf.random_normal_initializer(
                                     stddev=np.sqrt(1 / 256)), name="means")

        logvar_speed = (10 * 256) // 48
        self._log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(self._log_vars, axis=0) + self.lParameters['log_variance']

    def _policy_nn(self):
        """ 
        Fully connected neural net for policy approximation function
        """

        layers = self.networkParameters['layers_actor']

        # First hidden layer
        out = tf.layers.dense(self.obs_ph, layers[0], getattr(tf.nn, self.nParameters['activation_h_a']),
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")

        # Hidden layers
        for layer in layers[1:]:
            out = tf.layers.dense(out, layer, getattr(tf.nn, self.nParameters['activation_h_a']),
                                kernel_initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1 / layer)))
        # Output layer
        self.means = tf.layers.dense(out, self.act_dim,tf.nn.sigmoid,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / layers[-1])), name="means")
       
        logvar_speed = (10 * layers[-1]) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.lParameters['log_variance']

        print('Policy Params -- {}, lr: {:.3g}, logvar_speed: {}'
              .format(layers, self.lParameters['lr_actor'], logvar_speed))

    def _logprob(self):
        """
        Calculate log probabilities of a batch of observations & actions
        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """

        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

 
    def scaleVariance(self):
        """
        Scales the variance of the current policies 
        """
        with self.g.as_default() as gg:
            self.sampled_act = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)) * self.lParameters['variance_coeff']
            self.log_vars = tf.reduce_sum(self._log_vars, axis=0) + self.lParameters['log_variance']

    def _loss_train_op(self):
        """
        Loss function of the actor net
        """
        pg_ratio = tf.exp(self.logp - self.logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.lParameters['clipping'], 1 + self.lParameters['clipping'])
        surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                    self.advantages_ph * clipped_pg_ratio)
        self.loss = -tf.reduce_mean(surrogate_loss)

        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """
        Launch TensorFlow session and initialize variables
        """
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """
        Draw sample from policy distribution
        """
        feed_dict = {self.obs_ph: obs}
        
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """ 
        Update the policy network
        Returns the average loss during the epochs
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: 0,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lParameters['lr_actor'] * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss = []
        for _ in range(self.lParameters['epochs']):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss.append(self.sess.run([self.loss], feed_dict))
        return np.mean(loss)

    def close_sess(self):
        """
        Close TensorFlow session
        """
        self.sess.close()
        # Cr

    def save(self, date):
        """
        Save the tensorflow weights file
        """
        path = '{}_{}_POLICY.cpkt'.format('BP', date)
        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1)
        saver.save(self.sess, path)
        print('Trained model saved at {}'.format(path))

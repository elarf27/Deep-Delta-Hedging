import numpy as np
import tensorflow as tf
import h5py
from scipy import stats
from pathlib import Path



class StochasticProcess(object):
    def __init__(self, steps, paths, T, drift, diffusion, init):
        self.steps = steps
        self.paths = paths
        self.T = T 
        self.drift = drift
        self.diffusion = diffusion
        self.init = init

    def gen_process(self):
        Z = np.random.normal(0, 1, [self.paths, self.steps])
        S = np.zeros([self.paths, self.steps + 1])
        t = np.zeros([self.steps + 1])

        S[:, 0] = np.log(self.init)
        dt = self.T/ self.steps
        for step in range(0, self.steps):
            S[:, step + 1] = S[:, step] + (self.drift - .5 * self.diffusion ** 2) * dt + self.diffusion * np.sqrt(dt) * Z[:, step]
            t[step + 1] = t[step] + dt
        S_t = np.exp(S)
        expand = np.transpose(np.expand_dims(S_t, axis = -1), (1, 0, 2))
        result = {'time': t, 'S': expand, 'raw': S_t}
        return result
        
    def density_fx(self, i, j):
        return stats.lognorm.pdf(i, scale = np.exp(np.log(self.init) + (self.drift - .5 * self.diffusion ** 2) * j), s = np.sqrt(j) * self.diffusion)
    

class BlackScholes(object):
    def __init__(self, type, init, strike, drift, diffusion, time, T):
        self.type = type
        self.init = init
        self.strike = strike
        self.drift = drift
        self.diffusion = diffusion
        self.time = time
        self.T = T
        self.dt = T - time
        self.d1 = (np.log(init / strike) + (drift + .5 * np.power(diffusion, 2)) * self.dt) / (diffusion * np.sqrt(self.dt))

    def option_price(self):
         d2 = self.d1 - self.diffusion * np.sqrt(self.dt)
         if self.type == 'Call':
             return stats.norm.cdf(self.d1) * self.init - stats.norm.cdf(d2) * self.strike * np.exp(- self.drift * self.dt)
         elif self.type == 'Put':
             return stats.norm.cdf(- d2) * self.strike * np.exp(- self.drift * self.dt) - stats.norm.cdf(- self.d1) * self.init
         
    def option_delta(self):
         if self.type == 'Call':
              return stats.norm.cdf(self.d1)
         elif self.type == 'Put':
              return stats.norm.cdf(self.d1) - 1
         

class HedgeAgent(object):
    def __init__(self, steps, batch, cost = .005):
        tf.compat.v1.reset_default_graph()
        self.batch = batch
        self.input = tf.compat.v1.placeholder(tf.float32, [steps, batch, 1])
        self.strike = tf.compat.v1.placeholder(tf.float32, [batch])
        self.quantile = tf.compat.v1.placeholder(tf.float32)
        self.cost = cost
        self.neurons = [120, 70, 40, 1]

        train_dir = Path('Deep Hedging')
        run = 0
        log = train_dir / f'run_{run:01}'
        log.mkdir(parents = True, exist_ok = True)
        self.hdf_path = train_dir / 'DeepHedging.h5'
        self.writer = tf.summary.create_file_writer(log.as_posix())

        spot_T = self.input[-1, :, 0]
        spot_t = tf.unstack(self.input[:-1, :, :], axis = 0)
        dspot = self.input[1:, :, 0] - self.input[:-1, :, 0]

        lstm_cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(i) for i in self.neurons]
        network = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)
        
        self.option = tf.maximum(spot_T - self.strike, 0)

        self.strategy, state = tf.compat.v1.nn.static_rnn(network, spot_t, initial_state = network.zero_state(batch, tf.float32), dtype = tf.float32)
        self.strategy = tf.reshape(self.strategy, [steps - 1, batch])

        self.hedge = -self.option + tf.reduce_sum(dspot * self.strategy, axis = 0) - self.get_cost()

        CVaR, indices = tf.nn.top_k(-self.hedge, tf.cast((1 - self.quantile) * batch, tf.int32))
        CVaR = tf.reduce_mean(CVaR)

        self.train_op = tf.compat.v1.train.AdamOptimizer().minimize(CVaR)
        self.store = tf.compat.v1.train.Saver()

    def get_cost(self):
        delta = tf.concat([tf.zeros_like(self.strategy[:1, :]), self.strategy], axis = 0)
        transaction_costs = tf.reduce_sum(tf.abs(delta[1:] - delta[:-1]) * self.cost, axis = 0)
        return transaction_costs
    
    def execute_hedge(self, paths, strikes, alpha, session, epochs, is_training):
        batch = self.batch
        indices = np.arange(paths.shape[1])

        for epoch in range(epochs):
            pnls = []
            strategies = []
            if is_training:
                np.random.shuffle(indices)

            for i in range(int(paths.shape[1] / batch)):
                batch_indices = indices[i * batch: (i + 1) * batch]
                batch_paths = paths[:, batch_indices, :]
                feed_dict = {self.input: batch_paths, self.strike: strikes[batch_indices], self.quantile: alpha}

                if is_training:
                    _, pnl, strategy = session.run([self.train_op, self.hedge, self.strategy], feed_dict)
                else:
                    pnl, strategy = session.run([self.hedge, self.strategy], feed_dict)

                pnls.append(pnl)
                strategies.append(strategy)

            CVaR = np.mean(-np.sort(np.concatenate(pnls))[:int((1 - alpha) * paths.shape[1])])

            if is_training and epoch % 10 == 0:
                print('Epoch', epoch)
                self.store.save(session, str(self.hdf_path / 'run.ckpt'))

        self.store.save(session, str(self.hdf_path / 'run.ckpt'))
        return CVaR, np.concatenate(pnls), np.concatenate(strategies, axis = 1)
    
    def train(self, paths, strikes, alpha, epochs, chap):
        chap.run(tf.compat.v1.global_variables_initializer())
        self.execute_hedge(paths, strikes, alpha, chap, epochs, is_training = True)

    def predict(self, paths, strikes, alpha, chap):
        return self.execute_hedge(paths, strikes, alpha, chap, epochs = 1, is_training = False)
    
    def restore(self, chap, ckpt):
        self.store.restore(chap, str(self.hdf_path / ckpt))




        






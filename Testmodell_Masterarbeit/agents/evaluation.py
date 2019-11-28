"""
Datei enthält Klassen für das Evaluieren der Agenten
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import StateOperator


def get_statistik(env):
    """
    Takes the results out of the Simulation-Statistics Object
    :param env:
    :return:
    """
    artikels = env.possibles
    statistik = np.zeros((len(artikels), 3))
    for id_art, art_mkt in enumerate(artikels):
        aktueller_artikel = int(str(art_mkt)[-6:])
        abschriften = np.sum(env.statistics.abschrift(aktueller_artikel))
        fehlmenge = np.sum(env.statistics.fehlmenge(aktueller_artikel))
        reward = np.mean(env.statistics.rewards(aktueller_artikel))
        absatz = np.sum(env.statistics.absaetze(aktueller_artikel))
        actions = np.sum(env.statistics.actions(aktueller_artikel))
        if actions > 0:
            abschrift_quote = abschriften / actions
        else:
            abschrift_quote = 0
        statistik[id_art] = [reward, abschrift_quote, fehlmenge / absatz]

    return statistik


class Evaluator:
    """
    Evaluationsklasse
    """
    def __init__(self, agent, training_simulation, testing_simulation, hparams, session=None):
        self.agent = agent
        if hasattr(agent, 'act'):
            self.agent_type = 'DDDQN'
        else:
            self.agent_type = 'A3C'
        self.sess = session
        self.train_env = training_simulation
        self.test_env = testing_simulation
        self.name = str(hparams.warengruppe[0])
        self.run_id = hparams.run_id
        self.hparams = hparams
        if hparams.detail_warengruppe is not None:
            self.name += '-' + str(hparams.detail_warengruppe[0])

    def show(self):
        stats = self.run(self.train_env)
        self.plot(stats, path=os.path.join('files', 'graphics'), category='Train')
        stats = self.run(self.test_env)
        self.plot(stats, path=os.path.join('files', 'graphics'), category='Test')

    def run(self, env):
        """
        Method to run all possible episodes with the agent
        :return:
        """
        state_op = StateOperator(self.hparams)
        artikels = env.possibles
        if self.agent_type == 'DDDQN':
            for artikel in tqdm(artikels):
                state, info = env.reset(artikel)
                state_op.start(state)
                done = False
                while not done:
                    action = self.agent.act(state_op.state)
                    reward, done, next_state = env.make_action(np.argmax(action))
                    state_op.add(next_state)
        else:
            for artikel in tqdm(artikels):
                state, info = env.reset(artikel)
                state_op.start(state)
                done = False
                while not done:
                    strategy, value = self.sess.run(
                        [self.agent.policy, self.agent.value],
                        feed_dict={
                            self.agent.inputs: [state_op.state]
                        }
                    )
                    action = np.random.choice(strategy[0], p=strategy[0])
                    action = np.argmax(strategy == action)
                    reward, done, next_state = env.make_action(action)
                    state_op.add(next_state)
        stats = get_statistik(env)
        return stats

    def plot(self, statistik, path=None, reward_range=(0, 3), category=''):
        """
        Plots the results. If Path is specified, then Image is directly saved
        :param path:
        :param statistik: np.ndarray
        :param reward_range
        :param category
        :return:
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.set_title('{typ}-Agent im {cat}-Datensatz der Warengruppe {name}'.format(
            typ=self.agent_type,
            name=self.name,
            cat=category))
        ax1.hist(statistik[:, 0], bins=100, range=reward_range, label=r'$\emptyset$-Belohnung', color='orangered')
        ax1.legend()

        ax2.hist(statistik[:, 1], bins=100, range=(0, 1), label=r'$\emptyset$-Abschriften Quote', color='limegreen')
        ax2.set_ylabel('Anzahl Artikel')
        ax2.legend()

        ax3.hist(statistik[:, 2], bins=100, range=(0, 1), label=r'$\emptyset$-Fehlmengen Quote', color='dodgerblue')
        ax3.legend()

        if path:
            plt.savefig(
                os.path.join(
                    path,
                    '{typ}-Agent Eval {run_id}-{cat} Warengruppe {name}',
                ).format(typ=self.agent_type, run_id=self.run_id, cat=category, name=self.name))
        else:
            plt.show()

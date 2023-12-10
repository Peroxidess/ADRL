import copy
import itertools
import numpy as np
import pandas as pd
import torch
from model.TD3.TD3 import TD3


class RL():
    def __init__(self, state_dim, action_dim, seed, ckpt_dir, method_RL=''):
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.method_RL = method_RL
        if 'ADRL' in method_RL:
            self.agent = TD3(alpha=0.01, beta=0.005, gama=2e-3, state_dim=self.state_dim,
                        action_dim=self.action_dim, actor_fc1_dim=state_dim//4, actor_fc2_dim=state_dim//8,
                        critic_fc1_dim=state_dim//4, critic_fc2_dim=state_dim//8, ckpt_dir=self.checkpoint_dir, gamma=0.99,
                        tau=0.005, action_noise=0.01, policy_noise=0.01, policy_noise_clip=0.01,
                        delay_time=4, max_size=6000, batch_size=256, seed=seed)

    def store_data(self, data, label):
        self.agent.memory_init()
        action_mean = data.filter(regex=r'action').mean()
        action_std = data.filter(regex=r'action').std()
        id_unique = data['病例编号'].unique()
        for id_ in id_unique:
            data_single = data[data['病例编号'] == id_]
            label_single = label.loc[data_single.index]
            col_lv = data_single.filter(regex=r'次序').columns

            for e in itertools.product(set(data_single[col_lv[0]]), set(data_single[col_lv[1]])):
                data_single_ = data_single[(data_single[col_lv[0]] == e[0]) & (data_single[col_lv[1]] == e[1])]
                data_single_lv3 = data_single_.drop(columns=data_single.filter(regex=r'次序|编号').columns)
                done = False
                if data_single_lv3.shape[0] == 1:
                    print(f'id_')
                for index_ in range(data_single_lv3.shape[0] - 1):
                    if index_ == 0:
                        action_pre = pd.Series(0, index=['action'])
                        action_post = data_single.filter(regex=r'action').iloc[index_]
                    else:
                        action_pre += data_single.filter(regex=r'action').iloc[index_ - 1]
                        action_post += data_single.filter(regex=r'action').iloc[index_]

                    action = data_single_lv3.filter(regex=r'action').iloc[index_]
                    reward = 1 - label_single.iloc[index_]['label1']
                    observation = data_single_lv3.drop(columns=data_single.filter(regex=r'时间|label|action').columns).iloc[index_]
                    observation = pd.concat([observation, action_pre])

                    observation_ = data_single_lv3.drop(columns=data_single.filter(regex=r'时间|label|action').columns).iloc[index_ + 1]
                    observation_ = pd.concat([observation_, action_post])

                    if index_ == data_single_lv3.shape[0] - 2:
                        done = True
                        reward = 1 - label_single.iloc[index_]['label1']
                    self.agent.remember(observation.values, action.values, reward, observation_.values, done)

    def learn(self, val_x, val_label, episodes_RL):
        critic_loss_all = []
        actor_loss_all = []
        q_test_list = []
        q_test_df_mean_epoch = pd.DataFrame([])
        for episode in range(episodes_RL):
            Flag_training, critic_loss, actor_loss = self.agent.learn()
            critic_loss_all.append(critic_loss)
            actor_loss_all.append(actor_loss)
            q_test_df = self.eval(copy.deepcopy(val_x), val_label, type_task='RL')
            q_test_df_mean = q_test_df.drop(columns=['id']).mean(axis=0)
            q_test_df_mean_epoch = pd.concat([q_test_df_mean_epoch, pd.DataFrame(q_test_df_mean.values.T.reshape(1, -1), columns=q_test_df_mean.index)], axis=0)
            q_test_list.append(q_test_df)
            if not Flag_training:
                break
        return critic_loss_all, actor_loss_all, q_test_list, q_test_df_mean_epoch

    def eval(self, x, y, type_task='RL'):
        q_test_df = pd.DataFrame([])
        q_test_df['id'] = x['病例编号'].values
        x.drop(columns=x.filter(regex=r'次序|病例编号').columns, inplace=True)
        if 'RL' in type_task:
            q_test_record, q_test_prediction, q_zero_test, q_mean_test, q_random_test, action, action_diff = self.predict_q(x)
            q_test_df['label1'] = y['label1'].values
            q_test_df['q_record'] = q_test_record
            q_test_df['q'] = q_test_prediction
            q_test_df['q_zero'] = q_zero_test
            q_test_df['q_mean'] = q_mean_test
            q_test_df['q_random'] = q_random_test
            q_test_df['q_diff'] = q_test_prediction - q_test_record
            q_test_df['action_diff'] = action_diff
            q_test_df['action'] = action
            q_test_df['action_raw'] = q_test_df[['action']] - action_diff

        if 'C' in type_task:
            y_test_record, y_test_prediction, y_zero_test, y_mean_test, y_random_test = self.predict_y(x)
            q_test_df['label1'] = y['label1'].values
            q_test_df['y_test_record'] = y_test_record[:, 0]
            q_test_df['y_test_prediction'] = y_test_prediction[:, 0]
            q_test_df['y_recordact_diff'] = y['label1'].values - y_test_record[:, 0]
            q_test_df['y_predact_diff'] = y['label1'].values - y_test_prediction[:, 0]

        return q_test_df

    def predict(self, data):
        observation = data.drop(columns=data.filter(regex=r'时间|label|action').columns)
        observation_initaction = pd.concat(
            [observation, pd.Series(np.zeros(shape=observation.shape[0]), index=observation.index, name='action')],
            axis=1)
        action = self.agent.choose_action(observation_initaction.values, train=False)
        return action

    def predict_q(self, data):
        action_record = data.filter(regex=r'action')
        action_zero = np.zeros(action_record.shape)
        action_mean = np.ones(action_record.shape) * action_record.mean().values[0]
        action_random = np.random.uniform(low=action_record.min(), high=action_record.max(), size=action_record.shape)
        observation = data.drop(columns=data.filter(regex=r'时间|label|action').columns)
        observation_initaction = pd.concat([observation, pd.Series(np.zeros(shape=observation.shape[0]), index=observation.index, name='action')], axis=1)
        action = self.agent.choose_action(observation_initaction.values, train=False).reshape(-1, 1)
        q_record = self.agent.predict_q(observation_initaction.values, action_record.values)
        q_zero = self.agent.predict_q(observation_initaction.values, action_zero)
        q_mean = self.agent.predict_q(observation_initaction.values, action_mean)
        q_random = self.agent.predict_q(observation_initaction.values, action_random)
        q = self.agent.predict_q(observation_initaction.values, action)
        action_diff = action - action_record.values
        return q_record, q, q_zero, q_mean, q_random, action, action_diff

    def predict_y(self, data):
        np.random.seed(self.seed)
        data_copy = copy.deepcopy(data)
        action_record = data_copy.filter(regex=r'action')
        action_zero = np.zeros(action_record.shape)
        action_mean = np.ones(action_record.shape) * action_record.mean().values[0]
        action_random = np.random.normal(loc=action_record.mean(), scale=action_record.std(), size=action_record.shape)
        observation = data.drop(columns=data.filter(regex=r'时间|label|action').columns)
        observation_initaction = pd.concat(
            [observation, pd.Series(np.zeros(shape=observation.shape[0]), index=observation.index, name='action')],
            axis=1)
        action = self.agent.choose_action(observation_initaction.values, train=False).reshape(-1, 1)
        y_record = self.agent.predict_y(observation_initaction.values, action_record.values)
        y_RL = self.agent.predict_y(observation_initaction.values, action)
        y_zero = self.agent.predict_y(observation_initaction.values, action_zero)
        y_mean = self.agent.predict_y(observation_initaction.values, action_mean)
        y_random = self.agent.predict_y(observation_initaction.values, action_random)
        return y_record, y_RL, y_zero, y_mean, y_random

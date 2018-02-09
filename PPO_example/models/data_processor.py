from basic_utils.utils import *


class Ensemble:
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, paths):
        for p in self.processors:
            paths = p(paths)
        return paths


class Scale_Reward:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, paths):
        for path in paths:
            path['reward'] = path['reward'] * self.scale

        return paths


class Calculate_Return:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, paths):
        for path in paths:
            path['return'] = discount(path['reward'], self.gamma)

        return paths


class Predict_Value:
    def __init__(self, vf):
        self.vf = vf

    def __call__(self, paths):
        for path in paths:
            path['value'] = self.vf.predict(path["observation"]).reshape((-1,))

        return paths


class Calculate_Generalized_Advantage:
    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam

    def __call__(self, paths):
        for path in paths:
            rewards = path['reward']
            values = path['value']

            tds = rewards - values + np.append(values[1:] * self.gamma, 0)
            advantages = discount(tds, self.gamma * self.lam)
            path['advantage'] = advantages
        alladv = np.concatenate([path["advantage"] for path in paths])
        std = alladv.std()
        mean = alladv.mean()
        for path in paths:
            path["advantage"] = (path["advantage"] - mean) / std

        return paths


class Extract_Item_By_Name:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, paths):
        new_paths = []
        for path in paths:
            new_path = defaultdict(list)
            for k in self.keys:
                if k in path:
                    new_path[k] = path[k]
            new_paths.append(new_path)

        return new_paths


class Concatenate_Paths:
    def __init__(self):
        pass

    def __call__(self, paths):
        keys = paths[0].keys()
        new_path = defaultdict(list)
        for k in keys:
            new_path[k] = np.concatenate([path[k] for path in paths])
            new_path[k] = np_to_var(np.array(new_path[k]))
            if len(new_path[k].size()) == 1:
                new_path[k] = new_path[k].view(-1, 1)
        return new_path


class No_Process:
    def __init__(self):
        pass

    def __call__(self, paths):
        return paths


class Predict_Next_Action:
    def __init__(self, pol, target=True):
        self.pol = pol
        self.target = target

    def __call__(self, paths):
        for path in paths:
            next_observations = path['next_observation']
            path['next_action'] = self.pol.act(next_observations, target=self.target)
        return paths


class Calculate_Next_Q_Value_AS:
    def __init__(self, qf):
        self.qf = qf

    def __call__(self, paths):
        for path in paths:
            next_observations = path['next_observation']
            actions = path['next_action']
            path['next_q_value'] = self.qf.predict(next_observations, actions, target=True)
        return paths


class Calculate_Next_Q_Value:
    def __init__(self, qf, double):
        self.qf = qf
        self.double = double

    def __call__(self, paths):
        for path in paths:
            next_observations = path['next_observation']
            if not self.double:
                path['next_q_value'] = self.qf.predict(next_observations, target=True).max(axis=1)
            else:
                ty = self.qf.predict(next_observations).argmax(axis=1)
                path['next_q_value'] = self.qf.predict(next_observations, target=True)[np.arange(next_observations.shape[0]), ty]
        return paths


class Calculate_Q_Target:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, paths):
        for path in paths:
            not_dones = path['not_done']
            rewards = path['reward']
            next_q_values = path['next_q_value']
            path['y_targ'] = next_q_values.reshape((-1,)) * not_dones * self.gamma + rewards
        return paths

from models.agents import *
from basic_utils.data_generator import Parallel_Path_Data_Generator
from basic_utils.env_wrapper import Scaler


class Path_Trainer:
    def __init__(self,
                 agent,
                 env,
                 data_generator,
                 data_processor,
                 save_every=None,
                 print_every=10):
        self.callback = Callback()
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.data_generator = data_generator
        self.data_processor = data_processor

    def train(self):
        count = 1
        for paths, path_info, extra_info in self.data_generator():
            processed_path = self.data_processor(paths)
            u_stats, info = self.agent.update(processed_path)
            self.callback.add_update_info(u_stats)
            self.callback.add_path_info(path_info, extra_info)

            if self.callback.num_batches() >= self.print_every:
                count = self.callback.print_table()

            if self.save_every is not None and count % self.save_every == 0:
                self.agent.save_model('./save_model/' + self.env.name + '_' + self.agent.name)

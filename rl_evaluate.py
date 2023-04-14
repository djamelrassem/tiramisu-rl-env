import argparse, ray
from ray.rllib.models import ModelCatalog
from rl_agent.rl_env import TiramisuRlEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from rl_agent.rl_policy_lstm import PolicyLSTM
from config.config import Config, DatasetFormat
from rl_agent.rl_policy_nn import PolicyNN
from ray.air.checkpoint import Checkpoint
import numpy as np

from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor

parser = argparse.ArgumentParser()

parser.add_argument("--run",
                    type=str,
                    default="PPO",
                    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    Config.init()
    Config.config.dataset.is_benchmark = True

    dataset_actor = DatasetActor.remote(Config.config.dataset)

    match (Config.config.experiment.policy_model):
        case "lstm":
            ModelCatalog.register_custom_model("policy_nn", PolicyLSTM)
            model_custom_config = Config.config.lstm_policy.__dict__ 
        case "ff":
            ModelCatalog.register_custom_model("policy_nn", PolicyNN)
            model_custom_config = Config.config.policy_network.__dict__

    config = PPOConfig().framework(args.framework).environment(
        TiramisuRlEnv,
        env_config={
            "config": Config.config,
            "dataset_actor": dataset_actor
        }).rollouts(num_rollout_workers=1)
    config.explore = False

    config = config.to_dict()
    config["model"] = {
        "custom_model": "policy_nn",
        "vf_share_layers": Config.config.experiment.vf_share_layers,
        "custom_model_config": model_custom_config
    }

    checkpoint = Checkpoint.from_directory(Config.config.ray.restore_checkpoint)
    ppo_agent = PPO(AlgorithmConfig.from_dict(config))
    ppo_agent.restore(checkpoint_path=checkpoint)

    env = TiramisuRlEnv(config={
        "config": Config.config,
        "dataset_actor": dataset_actor
    })
    match (Config.config.experiment.policy_model):
        case "lstm":
            lstm_cell_size = model_custom_config["lstm_state_size"]
            init_state = state = [
                np.zeros([lstm_cell_size], np.float32) for _ in range(2)
            ]
            for i in range(31):
                observation, _ = env.reset()
                episode_done = False
                state = init_state
                while not episode_done:
                    action, state_out, _ = ppo_agent.compute_single_action(
                        observation=observation,
                        state=state,
                        explore=False,
                        policy_id="default_policy")
                    observation, reward, episode_done, _, _ = env.step(action)
                    state = state_out
                else:
                    env.tiramisu_api.final_speedup()
                    print()
        case "ff":
            for i in range(31):
                observation, _ = env.reset()
                episode_done = False
                while not episode_done:
                    action = ppo_agent.compute_single_action(observation=observation,
                                                            explore=False,policy_id="default_policy")
                    observation, reward, episode_done, _, _ = env.step(action)
                else:
                    env.tiramisu_api.final_speedup()
                    print()

    ray.shutdown()
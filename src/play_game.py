import gym
from gym.utils.play import play
import csv
import os
from PIL import Image
import numpy as np


def save_sas_during_gameplay(env_name, dataset):
    """
    Allows manual gameplay and saves state-action-next_state data to a CSV file.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    if "render_fps" not in env.metadata or env.metadata["render_fps"] is None:
        env.metadata["render_fps"] = 10

    with open(dataset, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "next_state", "done"])  # Header row

        def callback(obs_t, obs_tp1, action, reward, terminated, truncated, info):
            obs_t = obs_t[0] if isinstance(obs_t, tuple) else obs_t
            obs_tp1 = obs_tp1[0] if isinstance(obs_tp1, tuple) else obs_tp1

            state_dir = "../data/states/"
            os.makedirs(state_dir, exist_ok=True)

            step_count = len(os.listdir(state_dir)) // 2  # Assuming two files per step
            state_file = os.path.join(state_dir, f"state_{step_count}_obs_t.png")
            next_state_file = os.path.join(state_dir, f"state_{step_count}_obs_tp1.png")

            Image.fromarray(np.uint8(obs_t)).resize((80, 80)).save(state_file)
            Image.fromarray(np.uint8(obs_tp1)).resize((80, 80)).save(next_state_file)

            writer.writerow(
                [state_file, action, next_state_file, reward, terminated or truncated]
            )

        print("Starting manual gameplay. Use the keys to control the agent.")
        print("Press 'ESC' to end the session and save the data.")

        play(env, callback=callback, fps=10, zoom=3)
    env.close()

    print(f"SAS dataset saved to {dataset}")


if __name__ == "__main__":
    dataset = "../data/dataset.csv"
    env_name = "ALE/MsPacman-v5"
    save_sas_during_gameplay(env_name, dataset)

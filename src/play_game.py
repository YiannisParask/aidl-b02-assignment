import gym
from gym.utils.play import play
import csv
import os
from PIL import Image
import numpy as np
import time


def save_sas_during_gameplay(env_name, dataset, write_header=False):
    """
    Allows manual gameplay and saves state-action-next_state data to a CSV file.
    
    Args:
        env_name: Name of the environment.
        dataset: Path to the CSV file to save the state-action-next_state data.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    if "render_fps" not in env.metadata or env.metadata["render_fps"] is None:
        env.metadata["render_fps"] = 10

    if write_header:
        with open(dataset, "a", newline="") as f:
            writer = csv.writer(f)
            
            # Check if the file is empty, if so, write the header row
            if os.stat(dataset).st_size == 0:
                writer.writerow(["state", "action", "next_state", "done"])  # Header row

            def callback(obs_t, obs_tp1, action, reward, terminated, truncated, info):
                obs_t = obs_t[0] if isinstance(obs_t, tuple) else obs_t
                obs_tp1 = obs_tp1[0] if isinstance(obs_tp1, tuple) else obs_tp1

                state_dir = "../data/states/"
                os.makedirs(state_dir, exist_ok=True)

                #step_count = len(os.listdir(state_dir)) // 2 
                timestamp = int(time.time() * 1000)
                state_file = os.path.join(state_dir, f"state_{timestamp}_obs_t.png")
                next_state_file = os.path.join(state_dir, f"state_{timestamp}_obs_tp1.png")

                Image.fromarray(np.uint8(obs_t)).resize((80, 80)).save(state_file)
                Image.fromarray(np.uint8(obs_tp1)).resize((80, 80)).save(next_state_file)

                writer.writerow(
                    [state_file, action, next_state_file, terminated or truncated]
                )  # removed reward as it is not needed in my current task

            print("Starting manual gameplay. Use the 'WASD' to control the agent.")
            print("Press 'ESC' to end the session and save the data.")

            play(env, callback=callback, fps=10, zoom=3)
    else:
        print("Dataset creation is disabled. Starting manual gameplay without recording.")
        print("Use the 'WASD' to control the agent.")
        print("Press 'ESC' to end the session.")
        play(env, fps=10, zoom=3)
        
    env.close()

    if write_header:
        print(f"SAS dataset saved to {dataset}")
    else:
        print("Gameplay session completed without saving data.")


if __name__ == "__main__":
    dataset = "../data/dataset.csv"
    env_name = "FreewayDeterministic-v4"
    write_header = False
    save_sas_during_gameplay(env_name, dataset, write_header)

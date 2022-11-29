import gym
import tensorboard
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import highway_env

# Define the training environment
def train_env():
    # make a highway-fast environment
    env = gym.make('highway-fast-v0')
    #
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",       # Set Grayscale Image as changed type
            "observation_shape": (128, 64),       # Set width as 128 and set 64 as length
            "stack_size": 4,                      # Set 4 images stacked
            "weights": [0.2989, 0.5870, 0.1140],  # Weights for RGB conversion
            "scaling": 1.75,
        },
    })
    env.reset()      # Initiate a new episode and return observation and information
    return env


# Define the test environment
def test_env():
    env = train_env()
    # Make the environment configurable
    env.configure({"policy_frequency": 15, "duration": 20 * 15})
    env.reset()
    return env


if __name__ == '__main__':
    # Create the model
    model = DQN('CnnPolicy', DummyVecEnv([train_env]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="highway_cnn/")
    # Return a trained model
    model.learn(total_timesteps=int(2e4))
    # Save all the attributes of the object and the model parameters in a zip-file.
    model.save("highway_cnn/model")
    # Record video
    model = DQN.load("highway_cnn/model")


    # Make mulitple test environments into a single environment
    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(env, "highway_cnn/videos/",
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="dqn-agent")
    obs = env.reset()
    for _ in range(video_length + 1):
        # Predict
        action, _ = model.predict(obs)
        # Take an action, compute the state and return 4-tuple
        obs, _, _, _ = env.step(action)
    env.close()

# export_to_tensorrt.py

import torch
import torch_tensorrt
from stable_baselines3 import PPO
from multi_agent_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue, SingleAgentWrapper

def export_model_to_tensorrt(model_path, agent_id, export_path):
    """
    Exports a trained SB3 PPO model to TensorRT.

    Parameters:
    - model_path: Path to the trained SB3 model.
    - agent_id: Identifier of the agent.
    - export_path: Path to save the TensorRT optimized model.
    """
    # Load the trained SB3 model
    model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Define the environment for tracing
    env = MultiRobotEnvSB3CollabTrue(
        num_robots=4,
        num_tasks=6,
        robots_per_task=2,
        field_size=15.0,
        max_episode_steps=300,
        collision_penalty=False,
        completion_radius=0.5,
        robot_radius=0.3,
        time_penalty=-0.1,
        max_velocity=1.0,
        max_angular_velocity=1.0,
    )

    # Create a single-agent wrapper for the specific agent
    single_agent_env = SingleAgentWrapper(env, agent_id)

    # Reset the environment to get initial observation
    obs = single_agent_env.reset()

    # Convert observation to tensor
    example_input = torch.from_numpy(obs).float().unsqueeze(0).cuda()

    # Convert the SB3 policy to a PyTorch script module
    policy = model.policy.to('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval()

    with torch.no_grad():
        traced_model = torch.jit.trace(policy, example_input)

    # Convert the traced model to Torch-TensorRT
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input((1, *obs.shape))],
        enabled_precisions={torch.float},  # Run with FP32
    )

    # Save the optimized model
    trt_model.save(export_path)
    print(f"TensorRT optimized model saved to {export_path}")

# Example usage:
if __name__ == "__main__":
    # Paths
    trained_model_path = "./checkpoints_true_multi_agent/ppo_multi_agent_final"
    agent_id = "robot_0"  # Change as needed
    export_path = "./trt_models/ppo_multi_agent_robot_0_trt.pt"

    # Export the model
    export_model_to_tensorrt(trained_model_path, agent_id, export_path)

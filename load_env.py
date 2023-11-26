from dotenv import load_dotenv
import os

load_dotenv("./env")
class ENV:
    WANDB_PROJECT_NAME = os.getenv('WANDB_PROJECT_NAME')
    WANDB_USER_NAME = os.getenv('WANDB_USER_NAME')

env_vars = ENV()

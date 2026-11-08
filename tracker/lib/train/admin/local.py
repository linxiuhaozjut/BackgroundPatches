from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        root = Path(__file__).resolve().parents[5]
        self.workspace_dir = str(root / 'outputs/tracker')
        self.tensorboard_dir = str(root / 'outputs/tracker/tensorboard')
        self.pretrained_networks = str(root / 'weights/tracker')

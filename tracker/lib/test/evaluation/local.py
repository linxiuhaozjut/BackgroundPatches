from pathlib import Path
from .environment import EnvSettings


def local_env_settings():
    project = Path(__file__).resolve().parents[4]
    root = project.parent
    settings = EnvSettings()
    settings.prj_dir = str(project)
    settings.save_dir = str(root / 'outputs/tracker')
    settings.results_path = str(root / 'outputs/tracker/results')
    settings.network_path = str(root / 'weights/tracker')
    settings.result_plot_path = str(root / 'outputs/tracker/plots')
    return settings

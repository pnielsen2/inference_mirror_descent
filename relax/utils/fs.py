import os
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).parent.parent.parent
PACKAGE_ROOT = Path(__file__).parent.parent

TEMP_ROOT = Path(tempfile.gettempdir())

# Shared wandb namespace. Both training (wandb.init) and analysis
# (api.runs / api.run) resolve the entity/project through these so a whole
# lab can log to and read from one place. Override per-user via the
# WANDB_ENTITY / WANDB_PROJECT env vars (e.g. after migrating to a wandb
# Team). The default keeps the historical personal entity so existing runs
# stay reachable and invited collaborators land in the same project.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "pnielsen2-harvard")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "diffusion_online_rl")


def wandb_entity_project() -> str:
    """``"<entity>/<project>"`` path used by the wandb public API."""
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}"


# Base dir for offline wandb run dirs, per-user on shared netscratch so lab
# members can read each other's runs in place (the Lab/ tree is group-readable
# for kdbrantley_lab). Override with WANDB_OFFLINE_BASE.
WANDB_OFFLINE_BASE = Path(
    os.environ.get(
        "WANDB_OFFLINE_BASE",
        f"/n/netscratch/kdbrantley_lab/Lab/{os.environ.get('USER', '')}/wandb",
    )
)

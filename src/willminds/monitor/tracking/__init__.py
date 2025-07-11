

def get_tracking(config):
    match config.tracking.type:
        case "swanlab":
            import swanlab
            from swanlab.integration.transformers import SwanLabCallback
            swanlab.init(**config.tracking)
            return swanlab, SwanLabCallback()
        case "wandb":
            import wandb
            wandb.init(
                entity=config.tracking.get("workspace", None),
                project=config.tracking.get("project", None),
                name=config.tracking.get("experiment_name", None),
                notes=config.tracking.get("description", None),
                config=config.tracking.get("config", None),
                dir=config.tracking.get("logdir", None),
                mode=config.trakcing.get("mode", "online")
            )
            return wandb, None
        case None:
            return None, None
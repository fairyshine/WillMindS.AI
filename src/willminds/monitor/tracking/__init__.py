

def get_tracking(config):
    match config.tracking.type:
        case "swanlab":
            import swanlab
            swanlab.init(**config.tracking)
            return swanlab
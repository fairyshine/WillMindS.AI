from willminds import config, logger
from willminds.utils import backup_files

def main():
    # get config
    exp_name = config.experiment
    # logging
    logger.info("Test the Monitor")
    logger.info(config.train.lr)
    config.seed = 5e-6
    logger.info(config.train.lr)
    train_args = config.train
    logger.info(config.model.hidden_dim)

if __name__ == "__main__":
		import os
		backup_files("src/",  # backup your code
               ["src/test"], # exclude this dir
                 os.path.join(config.output_dir,"src_backup")) # backup path
		main()
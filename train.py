from utils.set_up import set_up
import logging

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')


if __name__ == "__main__":
    main()
    

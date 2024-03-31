from housing.pipeline.pipeline import Pipeline
from housing.exception import HousingException
from housing.logger import logging
import os
from housing.config.configuration import Configuration

def main( ):
    try: 
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_path))
        pipeline.start()
        logging.info("Main function execution completed")
    
    except Exception as e:
        logging.error(f"{e}")
      


if __name__ == "__main__" :
    main()

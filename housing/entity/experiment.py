class Experiment :
    running_status = False
    """    
    This code enforces a singleton pattern for the Experiment class. 
    It ensures that only one instance of Experiment can exist at a time. 
    Any attempt to create a second instance while one is already running will 
    result in an exception.
    """    
    def __new__(cls , *args,**kwargs):
        if Experiment.running_status :
            raise Exception("Exception is already running hence new experiment can not be created")
        
        return super(Experiment,cls).__new__(cls,*args,**kwargs)
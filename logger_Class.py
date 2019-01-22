

class logClass():

#----------------------------------------------------------------------------------------------------------------------

# DEBUG : detailed information , typically of interest only when diagnosing problems

# INFO: Confirmation that things are working as expected

# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (eg: ‘disk space low’). 
# The software is still working as expected

# ERROR: Due to a more serious problem, the software has not been able to perform some function.

# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

#----------------------------------------------------------------------------------------------------------------------

    import logging
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    handler = RotatingFileHandler('BrainTumour.log', mode='a', maxBytes=5*1024*1024, 
    backupCount=2, encoding=None, delay=0)

    handler.setLevel(logging.DEBUG)

    # create a logging "format time-name-level-message"
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)


'''
How to use?

If you want to log anything to the generated log file called BrainTumour.log type :
from logger_Class import logClass

logClass.logger.debug("message")
logClass.logger.info("message")
logClass.logger.warning("message")
logClass.logger.error("message")
logClass.logger.critical("message")

Remember to keep an Alias for logClass.logger

Example result in BrainTumour.log
2019-01-21 21:31:38,318 - logger_Class - DEBUG - Hello

The log file will be updated as Time:Name:Level:Message format

Change logging.Formatter if you want any other format 

link to python logging documentation https://docs.python.org/3/library/logging.html

'''

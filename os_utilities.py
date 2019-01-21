import shutil, os, logging, traceback

def ShiftFolder(src:str, dst: str) -> bool:
    '''
    Function to shift a entire folder to a new
    location with all its contents
    
    Arguments:
        src {str} -- source of the folder
        dst {str} -- destination of the folder
    
    Returns:
        bool -- status check
    '''
    try:
        shutil.copytree(src = src, dst = dst)

    except Exception as err:
        print("[FATAL]:Cannot shift folders -- ", err)
        print(traceback.print_exc())
        return False

    finally:
        print("[INFO]: Moved Folder Successfully")
    
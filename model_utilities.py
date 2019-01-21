from keras import backend as K
from keras.models import Model, model_from_json
import os, time, traceback

def SaveModelJSON(model: Model, modelName: str,save_dir = "") -> bool:
    '''
    Function to save the model to a JSON format with
    weights as h5py for later loading and referencing
    
    Arguments:
        model {Model} -- Keras trained model
        modelName {str} -- name of the model
    Keyword Arguments:
        save_dir {str} -- path to save the model (default: {""})
    
    Returns:
        bool -- status check
    '''

    import logging
    modelJSON = model.to_json()

    try:
        if os.path.exists(save_dir + modelName + ".json"):
            raise FileExistsError 
        else:
            with open(save_dir + modelName + ".json", "w") as f:
                f.write(modelJSON)
            # Save weights
            model.save_weights(save_dir + modelName + ".h5")
            print("\n [+] Model saved in disk")
            return True
    except FileExistsError:
        logging.error(FileExistsError)
        print("\n [-] Warning file already exists at path " + save_dir + modelName + ".json")
        choice = input("[\n+] Do you want to overwrite the file? ")
        choice = choice.lower()
        if(choice == 'y'):
            print("\n [+] Overwriting... ")
            with open(save_dir + modelName + ".json", "w") as f:
                f.write(modelJSON)
            # Save weights
            model.save_weights(save_dir + modelName + ".h5")
            print("\n [+] Model saved in disk")
            return True
        else:
            NewModelName = input("\n [+] Enter new model name: ")
            NewDIR = input("\n [+] Enter new path: ")
            SaveModelJSON(model, NewModelName, NewDIR)
    finally:
        f.close()
        return False

def LoadModelJSON(modelName:str, load_dir = "") -> Model:
    '''
    Function to load the JSON model and return it
    
    Arguments:
        modelName {str} -- name of the model to load
    
    Keyword Arguments:
        load_dir {str} -- The model's saved directory (default: {""})
    
    Returns:
        Model -- the trained keras model -- need to compile!!!
    '''

    import logging

    try:
        if os.path.exists(load_dir + modelName + ".json"):
            jsonFile = open(load_dir + modelName + ".json", "r")
            loadedModelJson = jsonFile.read()
            jsonFile.close()
            model = model_from_json(loadedModelJson)
            model.load_weights(load_dir + modelName + ".h5")
            print("\n [+] Model loaded. Please compile it")
            return model
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        choice = input("\n [+] File not found. Do you want to enter a new DIR (Y) or exit (n) ?")
        choice = choice.lower()
        if(choice == 'y'):
            NewModelName = input("\n [+] Enter new model name: ")
            NewDIR = input("\n [+] Enter new path to load: ")
            print("Path selected : " +  NewDIR + NewModelName + ".json")
            LoadModelJSON(NewModelName, NewDIR)
        else:
            print("\n [+] Exitting ...")
            exit()
    finally:
        jsonFile.close()
        return None

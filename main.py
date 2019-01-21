import UNet
import numpy as np
from loss_utilities import dice_coef_loss, dice_coef, jaccard_distance_loss

if __name__ == "__main__":

    # Get Data
    import dataset

    # Assign it
    XTrain = dataset.X_train_input
    yTrain = dataset.X_train_target
    XDev = dataset.X_dev_input
    yDev = dataset.X_dev_target
    XTest = dataset.X_test_input
    yTest = dataset.X_test_target
    print("[INFO] Data Loaded")

    # Setup Data
    yTrain = (yTrain > 0).astype(int)
    yDev = (yDev > 0).astype(int)
    yTest = (yTest > 0).astype(int)
    X = np.asarray(XTrain[80])
    y = np.asarray(yTrain[80])
    nw, nh, nz = X.shape

    # Hyperparameter settings
    batch_size = 5
    lr = 0.0001
    n_epoch = 10

    # Initialize model
    model = UNet.UNet(shape = (batch_size, nw, nh, nz), dropout = 0.05, batchNorm = True)
    model.SetHyperParameters(learning_rate = lr, batchSize = batch_size, loss = dice_coef, metrics = [dice_coef, jaccard_distance_loss])
    model.SetupCallbacks()
    model.CompileModel()
    
    # Train model
    model.Train(XTrain, yTrain, XDev, yDev, n_epoch = n_epoch)

    # Eval model
    model.Evaluate(XTest, yTest)

    # pred model
    model.Predict(XTest, yTest)

    # Save model
    model.SaveModel()
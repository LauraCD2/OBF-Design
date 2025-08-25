def classify(model, train_dataset, test_dataset, save_name=None):

    mse_train, mae_train, mape_train, r2_train, datatrain = model.test(train_dataset)
    mse_test, mae_test, mape_test, r2_test, datatest   = model.test(test_dataset)


    dict_metrics = {
        "train": {"MSE": mse_train, "MAE": mae_train, "MAPE": mape_train, "R2": r2_train},
        "test": {"MSE": mse_test, "MAE": mae_test, "MAPE": mape_test, "R2": r2_test},
    }

    outs = [ datatrain[0], datatrain[1], datatest[0], datatest[1]]

    return dict_metrics, outs


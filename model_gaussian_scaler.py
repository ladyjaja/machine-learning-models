from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA
import colour_predict_func as colour_predict


def main():

    X_train, X_valid, y_train, y_valid = colour_predict.data()
    
    modelrgb = make_pipeline(
            MinMaxScaler(),
            GaussianNB()
            )
    modelrgb.fit(X_train, y_train)
    print("Model Score RGB Training: ", modelrgb.score(X_train, y_train))
    print("Model Score RGB Validation: ", modelrgb.score(X_valid, y_valid))
    #colour_predict.plot_predictions(modelrgb)
    
    modellab = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_lab, validate=False),
            MinMaxScaler(),
            GaussianNB()
            )
    modellab.fit(X_train, y_train)
    print("Model Score LAB Training: ", modellab.score(X_train, y_train))
    print("Model Score LAB Validation: ", modellab.score(X_valid, y_valid))
    #colour_predict.plot_predictions(modellab)
    
    modelhsv = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_hsv, validate=False),
            MinMaxScaler(),
            GaussianNB()
            )
    modelhsv.fit(X_train, y_train)
    print("Model Score HSV Training: ", modelhsv.score(X_train, y_train))
    print("Model Score HSV Validation: ", modelhsv.score(X_valid, y_valid))
    #colour_predict.plot_predictions(modelhsv)
    
    modelall = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            MinMaxScaler(),
            GaussianNB()
            )
    modelall.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelall.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelall.score(X_valid, y_valid))
    #colour_predict.plot_predictions(modelall)
    
    modelallpca = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            MinMaxScaler(),
            PCA(),
            GaussianNB()
            )
    modelallpca.fit(X_train, y_train)
    print("Model Score All Features with PCA Training: ", modelallpca.score(X_train, y_train))
    print("Model Score All Features with PCA Validation: ", modelallpca.score(X_valid, y_valid))
    #colour_predict.plot_predictions(modelallpca)

    
#    print(X_train.shape, X_valid.shape)
#    print(y_train.shape, y_valid.shape)
    
#    model = GaussianNB()
#    model.fit(X_train, y_train)
#    y_predicted = model.predict(X_valid)
#    print(y_predicted.shape)
#    colour_predict.plot_predictions(model)
    
#    print("Model Score Training")
#    print(model.score(X_train, y_train))
#    print("Model Score Validation")
#    print(model.score(X_valid, y_valid))
#    print("Classification Report")
#    print(classification_report(y_valid, model.predict(X_valid)))
    
    
if __name__ == '__main__':
    main()


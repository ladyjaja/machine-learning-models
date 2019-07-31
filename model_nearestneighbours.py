from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
import colour_predict_func as colour_predict


def main():
    X_train, X_valid, y_train, y_valid = colour_predict.data()
    k = 10
    
    modelrgb = make_pipeline(
           KNeighborsClassifier(n_neighbors=k)
            )
    modelrgb.fit(X_train, y_train)
    print("Model Score RGB Training: ", modelrgb.score(X_train, y_train))
    print("Model Score RGB Validation: ", modelrgb.score(X_valid, y_valid))
    
    modellab = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_lab, validate=False),
            KNeighborsClassifier(n_neighbors=k)
            )
    modellab.fit(X_train, y_train)
    print("Model Score LAB Training: ", modellab.score(X_train, y_train))
    print("Model Score LAB Validation: ", modellab.score(X_valid, y_valid))
    
    modelhsv = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_hsv, validate=False),
            KNeighborsClassifier(n_neighbors=k)
            )
    modelhsv.fit(X_train, y_train)
    print("Model Score HSV Training: ", modelhsv.score(X_train, y_train))
    print("Model Score HSV Validation: ", modelhsv.score(X_valid, y_valid))
    
    modelall = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            KNeighborsClassifier(n_neighbors=k)
            )
    modelall.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelall.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelall.score(X_valid, y_valid))
    
    modelallpca = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            PCA(),
            KNeighborsClassifier(n_neighbors=k)
            )
    modelallpca.fit(X_train, y_train)
    print("Model Score All Features with PCA Training: ", modelallpca.score(X_train, y_train))
    print("Model Score All Features with PCA Validation: ", modelallpca.score(X_valid, y_valid))
    
    modelallpoly = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            PolynomialFeatures(degree=4, include_bias=True),
            KNeighborsClassifier(n_neighbors=k)
            )
    modelallpoly.fit(X_train, y_train)
    print("Model Score All Features with Polynomial Training: ", modelallpoly.score(X_train, y_train))
    print("Model Score All Features with Polynomial Validation: ", modelallpoly.score(X_valid, y_valid))
    
#    model = KNeighborsClassifier(n_neighbors=k)
#    model.fit(X_train, y_train)
#    colour_predict.plot_predictions(model)
#    
#    print("Model Score Training")
#    print(model.score(X_train, y_train))
#    print("Model Score Validation")
#    print(model.score(X_valid, y_valid))
#    print("Classification Report")
#    print(classification_report(y_valid, model.predict(X_valid)))
    
    
if __name__ == '__main__':
    main()# -*- coding: utf-8 -*-


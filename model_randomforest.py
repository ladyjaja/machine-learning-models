from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
import colour_predict_func as colour_predict


def main():

    X_train, X_valid, y_train, y_valid = colour_predict.data()
    est = 400
    depth = 10
    leaf = 10
    
    modelrgb = make_pipeline(
         RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_leaf=leaf)
         )
    modelrgb.fit(X_train, y_train)
    print("Model Score RGB Training: ", modelrgb.score(X_train, y_train))
    print("Model Score RGB Validation: ", modelrgb.score(X_valid, y_valid))
    
    modellab = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_lab, validate=False),
            RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_leaf=leaf)
            )
    modellab.fit(X_train, y_train)
    print("Model Score LAB Training: ", modellab.score(X_train, y_train))
    print("Model Score LAB Validation: ", modellab.score(X_valid, y_valid))
    
    modelhsv = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_hsv, validate=False),
            RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_leaf=leaf)
            )
    modelhsv.fit(X_train, y_train)
    print("Model Score HSV Training: ", modelhsv.score(X_train, y_train))
    print("Model Score HSV Validation: ", modelhsv.score(X_valid, y_valid))
    
    modelall = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_leaf=leaf)
            )
    modelall.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelall.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelall.score(X_valid, y_valid))
    
    modelallpca = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            PCA(),
            RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_leaf=leaf)
            )
    modelallpca.fit(X_train, y_train)
    print("Model Score All Features with PCA Training: ", modelallpca.score(X_train, y_train))
    print("Model Score All Features with PCA Validation: ", modelallpca.score(X_valid, y_valid))
    
    
if __name__ == '__main__':
    main()# -*- coding: utf-8 -*-


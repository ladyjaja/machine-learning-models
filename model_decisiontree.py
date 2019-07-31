from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
import colour_predict_func as colour_predict


def main():
    X_train, X_valid, y_train, y_valid = colour_predict.data()
    depth = 8
    leaf = 6
    
    modelrgb = make_pipeline(
          DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modelrgb.fit(X_train, y_train)
    print("Model Score RGB Training: ", modelrgb.score(X_train, y_train))
    print("Model Score RGB Validation: ", modelrgb.score(X_valid, y_valid))
    
    modellab = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_lab, validate=False),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modellab.fit(X_train, y_train)
    print("Model Score LAB Training: ", modellab.score(X_train, y_train))
    print("Model Score LAB Validation: ", modellab.score(X_valid, y_valid))
    
    modelhsv = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_hsv, validate=False),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modelhsv.fit(X_train, y_train)
    print("Model Score HSV Training: ", modelhsv.score(X_train, y_train))
    print("Model Score HSV Validation: ", modelhsv.score(X_valid, y_valid))
    
    modelall = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modelall.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelall.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelall.score(X_valid, y_valid))
    
    modelallpca = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            PCA(),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modelallpca.fit(X_train, y_train)
    print("Model Score All Features with PCA Training: ", modelallpca.score(X_train, y_train))
    print("Model Score All Features with PCA Validation: ", modelallpca.score(X_valid, y_valid))
    
    modelallpoly = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            PolynomialFeatures(degree=4, include_bias=True),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            )
    modelallpoly.fit(X_train, y_train)
    print("Model Score All Features with Polynomial Training: ", modelallpoly.score(X_train, y_train))
    print("Model Score All Features with Polynomial Validation: ", modelallpoly.score(X_valid, y_valid))
    
    
    #print(X_train.shape, X_valid.shape)
    #print(y_train.shape, y_valid.shape)
    
#    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4)
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
    main()
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
import colour_predict_func as colour_predict


def main():

    X_train, X_valid, y_train, y_valid = colour_predict.data()
    
    modelrgb = make_pipeline(
          MinMaxScaler(),    
          VotingClassifier([
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier(3)),
            ('svm', SVC(kernel='rbf', C=10, gamma=5)),
            ('tree1', DecisionTreeClassifier(max_depth=10)),
            ('tree2', DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)),])
            )
    modelrgb.fit(X_train, y_train)
    print("Model Score RGB Training: ", modelrgb.score(X_train, y_train))
    print("Model Score RGB Validation: ", modelrgb.score(X_valid, y_valid))
    
    modellab = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_lab, validate=False),
            MinMaxScaler(),  
            VotingClassifier([
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(3)),
                ('svm', SVC(kernel='rbf', C=10, gamma=5)),
                ('tree1', DecisionTreeClassifier(max_depth=10)),
                ('tree2', DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)),])
            )
    modellab.fit(X_train, y_train)
    print("Model Score LAB Training: ", modellab.score(X_train, y_train))
    print("Model Score LAB Validation: ", modellab.score(X_valid, y_valid))
    
    modelhsv = make_pipeline(
            FunctionTransformer(colour_predict.rgb_to_hsv, validate=False),
            MinMaxScaler(),  
            VotingClassifier([
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(3)),
                ('svm', SVC(kernel='rbf', C=10, gamma=5)),
                ('tree1', DecisionTreeClassifier(max_depth=10)),
                ('tree2', DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)),])
            )
    modelhsv.fit(X_train, y_train)
    print("Model Score HSV Training: ", modelhsv.score(X_train, y_train))
    print("Model Score HSV Validation: ", modelhsv.score(X_valid, y_valid))
    
    modelall = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            MinMaxScaler(),  
            VotingClassifier([
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(3)),
                ('svm', SVC(kernel='rbf', C=10, gamma=5)),
                ('tree1', DecisionTreeClassifier(max_depth=10)),
                ('tree2', DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)),])
            )
    modelall.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelall.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelall.score(X_valid, y_valid))
    
    modelallpca = make_pipeline(
            FunctionTransformer(colour_predict.get_all_features, validate=False),
            MinMaxScaler(),  
            PCA(),
            VotingClassifier([
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(3)),
                ('svm', SVC(kernel='rbf', C=10, gamma=5)),
                ('tree1', DecisionTreeClassifier(max_depth=10)),
                ('tree2', DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)),])
            )
    modelallpca.fit(X_train, y_train)
    print("Model Score All Features Training: ", modelallpca.score(X_train, y_train))
    print("Model Score All Features Validation: ", modelallpca.score(X_valid, y_valid))
    
    
if __name__ == '__main__':
    main()
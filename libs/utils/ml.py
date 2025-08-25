from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter

from libs.metrics import compute_metric_params, overall_accuracy, average_accuracy, kappa

svm_config = dict(
    C=1e5,
    kernel='rbf',
    gamma=1.
)

rfc_config = dict(
    n_estimators=1000,
    max_depth=20
)

mlp_config = dict(
    solver='adam',
    max_iter=1000,
    alpha=1e-3,
)

knn_config = dict(
    n_neighbors=3
)

CLASSIFIERS = dict(
    svm=[SVC, svm_config],
    rfc=[RandomForestClassifier, rfc_config],
    mlp=[MLPClassifier, mlp_config],
    knn=[KNeighborsClassifier, knn_config]
)


def classify(method, train_dataset, test_dataset, save_name=None):
    print("Running classifier...")
    method.fit(train_dataset['X'], train_dataset['Y'])

    y_pred_train = method.predict(train_dataset['X'].astype('float32'))
    matrix_train = confusion_matrix(train_dataset['Y'], y_pred_train)
    AA_train, add_train, number_train = compute_metric_params(matrix_train)
    OA_train = overall_accuracy(matrix_train, number_train)
    AA_mean_train = average_accuracy(AA_train)
    Kappa_train = kappa(OA_train, matrix_train, add_train)

    y_test_pred = method.predict(test_dataset['X'].astype('float32'))
    matrix_test = confusion_matrix(test_dataset['Y'], y_test_pred)
    AA_test, add_test, number_test = compute_metric_params(matrix_test)
    OA_test = overall_accuracy(matrix_test, number_test)
    AA_mean_test = average_accuracy(AA_test)
    Kappa_test = kappa(OA_test, matrix_test, add_test)

    if save_name is not None:
        writer = SummaryWriter(f"results_estandarizacion_data_training_banda/{save_name}")
        writer.add_scalar("train/OA", OA_train)
        writer.add_scalar("train/AA_mean", AA_mean_train)
        writer.add_scalar("train/Kappa", Kappa_train)
        writer.add_scalar("test/OA", OA_test)
        writer.add_scalar("test/AA_mean", AA_mean_test)
        writer.add_scalar("test/Kappa", Kappa_test)

    return dict(
        train=dict(OA=OA_train, AA_mean=AA_mean_train, Kappa=Kappa_train),
        test=dict(OA=OA_test, AA_mean=AA_mean_test, Kappa=Kappa_test)
    )


def build_classifier(classifier_name):
    classifiers, config = CLASSIFIERS[classifier_name]
    return classifiers(**config)

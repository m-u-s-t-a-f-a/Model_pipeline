import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier  # brew install gcc@5 , pip install xgboost
from sklearn.model_selection import cross_val_score


def plt_confusion_matrix(df, label, pred, save_fig=False, output_path=None):
    cm = confusion_matrix(df[label], df[pred])
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum

    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(cm_perc, annot=True, annot_kws={"size": 20}, fmt='.0%', cmap="Blues", cbar=False)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show()


def performance_report(df, label, pred):
    report = pd.DataFrame(classification_report(df[label], df[pred], output_dict=True, zero_division=0))
    metrics = report.loc['precision':'f1-score', '1']
    accuracy = report['accuracy'][0]
    summary = metrics.append(pd.Series(accuracy, index=['accuracy']))

    print(summary.to_string())


def model_prediction(pipeline, x_data, y_data):
    results = pd.DataFrame(y_data)
    results['majority_class'] = 0
    results['preds'] = pipeline.predict(x_data)
    results['preds_prob'] = np.round(pipeline.predict_proba(x_data)[:, 1], 2)

    return pd.concat([results, x_data], axis=1)


def plt_class_distribution(pred_class, y_pred_prob, save_fig=False, output_path=None):
    df = pd.concat([pred_class, y_pred_prob], axis=1)
    df.columns = ['Pred_class', 'Pred_prob']

    for i in pred_class.unique():
        df1 = df[df['Pred_class'] == i]['Pred_prob']
        ax = sns.distplot(df1, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 1.5},
                          label='Pred = ' + str(i))

    ax.set(xlabel='Predicted probabilities', ylabel='Relative volume', title='Distribution of Predicted probablities')

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show()


def correlated_vars(df, x_cont, threshold=0.65):
    correlated_features = []
    correlation_matrix = df[x_cont].corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.append(colname)

    return list(set(correlated_features))


def benchmark_model(x_vars, y, cv_folds=7, save_fig=False, output_path=None):
    """Benchmark model performance with default parameters to gauge relative performance."""

    models = [LogisticRegression(random_state=0),
              DecisionTreeClassifier(random_state=0),  # LinearSVC()
              RandomForestClassifier(random_state=0),
              XGBClassifier()]

    # Transform x_variables
    scaler = StandardScaler().fit(x_vars.values)
    features = scaler.transform(x_vars.values)

    # Score models
    cv_df = pd.DataFrame(index=range(cv_folds * len(models)))
    entries = []

    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, y, scoring='accuracy', cv=cv_folds)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    means = cv_df.groupby('model_name').accuracy.mean()

    # Plot performance
    fig = plt.subplots(figsize=(12, 8))
    box_plot = sns.boxplot(x='accuracy', y='model_name', data=cv_df, palette='viridis')
    sns.stripplot(x='accuracy', y='model_name', data=cv_df,
                  size=10, jitter=True, edgecolor="gray", linewidth=2, palette='viridis')

    plt.ylabel('')
    plt.title('Model Benchmark')

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show(), print(means)


def plt_feature_importance(top_k, pipeline, df, save_fig=False, output_path=None):
    num_vars = pipeline.named_steps['preprocessor'].transformers_[0][2].tolist()
    cat_vars = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(
        df.select_dtypes(['object']).columns).tolist()
    features = num_vars + cat_vars

    importances = pipeline.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    new_indices = indices[:top_k]

    fig = plt.figure(figsize=(12, 6))

    plt.barh(range(top_k), importances[new_indices], color='steelblue', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Feature Importance (Top {})'.format(top_k))
    plt.xlabel('Relative Importance')

    plt.margins(0)
    plt.tight_layout()
    plt.gca().invert_yaxis()

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show()


def extract_model_coefficients(pipeline, x_vars):
    # Variable names
    numeric_vars_pipeline = pipeline.named_steps['preprocessor'].transformers_[0][2].tolist()
    categorical_var_pipeline = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[
        'onehot'].get_feature_names(x_vars.select_dtypes(['object']).columns).tolist()
    model_vars = numeric_vars_pipeline + categorical_var_pipeline

    # Coefficients
    coeffs = pipeline.named_steps['model'].coef_[0].tolist()

    # Odds
    odds = np.exp(coeffs)

    # Increase in prob
    i_ = odds / (odds + 1)

    # Variable type
    var_ = list(itertools.repeat('numeric', len(numeric_vars_pipeline))) + list(
        itertools.repeat('categorical', len(categorical_var_pipeline)))

    # Std magnitude of scaled vars
    s_ = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scale'].scale_.tolist() + list(
        itertools.repeat(0, len(categorical_var_pipeline)))

    # Mean of scaled vars
    m_ = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scale'].mean_.tolist() + list(
        itertools.repeat(0, len(categorical_var_pipeline)))

    # Variance of scaled vars
    v_ = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scale'].var_.tolist() + list(
        itertools.repeat(0, len(categorical_var_pipeline)))

    # Model intercept
    inter_ = list(itertools.repeat(pipeline.named_steps['model'].intercept_[0],
                                   (len(numeric_vars_pipeline) + len(categorical_var_pipeline))))

    df = pd.DataFrame(model_vars)
    df['Var_type'] = var_
    df['Intercept'] = inter_
    df['Standard_Coeff'] = coeffs
    df['Odds'] = odds
    df['Prob'] = i_
    df['Mean_for_scaling'] = m_
    df['Std_for_scaling'] = s_
    df['Variance_for_scaling'] = v_

    df.columns.values[0] = 'Var'

    df = df.sort_values('Standard_Coeff', ascending=False)

    return df


def evalBinaryClassifier(model, x, y, labels=['Positives', 'Negatives']):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    model : fitted scikit-learn model with predict_proba & predict methods
        and classes_ attribute. Typically LogisticRegression or
        LogisticRegressionCV

    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        in the data to be tested, and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target vector relative to x.

    labels: list, optional
        list of text labels for the two classes, with the positive label first

    Displays
    ----------
    3 Subplots

    Returns
    ----------
    F1: float
    '''
    # model predicts probabilities of positive class
    p = model.predict_proba(x)
    if len(model.classes_) != 2:
        raise ValueError('A binary class problem is required')
    if model.classes_[1] == 1:
        pos_p = p[:, 1]
    elif model.classes_[0] == 1:
        pos_p = p[:, 0]

    # FIGURE
    plt.figure(figsize=[15, 4])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y, model.predict(x))
    plt.subplot(131)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                     annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives',
                'False Negatives', 'True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': pos_p, 'target': y})
    plt.subplot(132)
    plt.hist(df[df.target == 1].probPos, density=True, bins=25,
             alpha=.5, color='green', label=labels[0])
    plt.hist(df[df.target == 0].probPos, density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0, 1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    # 3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y, p[:, 1])
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(fp_rates, tp_rates, color='green',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    # plot current decision point:
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)
    plt.show()
    # Print and Return the F1 score
    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
    )
    print(printout)
    return F1

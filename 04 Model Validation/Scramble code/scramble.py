from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import (
    get_scorer, accuracy_score, recall_score, precision_score,
    roc_auc_score, matthews_corrcoef, average_precision_score
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.stats
import copy

class Scrambler:
    def __init__(self, model, iterations=100):
        """初始化 Scrambler 类。
        
        参数:
        - model: 使用的模型实例。
        - iterations: 执行扰乱操作的迭代次数。
        """
        self.base_model = copy.deepcopy(model)
        self.iterations = iterations
        self.progress_bar = False

    def validate(self, X, Y, method="train_test_split", scoring="accuracy", cross_val_score_aggregator="mean", pvalue_threshold=0.05, cv_kfolds=5, as_df=False, validation_data=None, progress_bar=False):
        """验证模型性能，计算原始模型和扰乱模型的评价指标。
        
        参数:
        - X, Y: 数据集的特征和标签。
        - method: 验证方法，可以是 'train_test_split' 或 'cross_validation'。
        - scoring: 用于评价模型性能的指标名称。
        - cross_val_score_aggregator: 交叉验证得分的聚合方式。
        - pvalue_threshold: 用于显著性测试的 P 值阈值。
        - cv_kfolds: 交叉验证的折数。
        - as_df: 是否将结果作为 DataFrame 返回。
        - validation_data: 如果提供，将使用这些数据进行测试而不是分割 X, Y。
        - progress_bar: 是否显示进度条。
        """
        model_scorer = get_scorer(scoring)
        result = None

        if method == "train_test_split":
            if validation_data:
                X_train, Y_train = X, Y
                X_test, Y_test = validation_data
            else:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            result = self.__evaluate_model(X_train, Y_train, X_test, Y_test, model_scorer, progress_bar)
        elif method == "cross_validation":
            result = self.__evaluate_model(X, Y, X, Y, model_scorer, progress_bar, cross_val=True, cv_kfolds=cv_kfolds)
        
        # 根据 as_df 参数决定返回数据的格式
        if as_df:
            df = pd.DataFrame(result)
            # 计算每个指标的 Z 分数、P 值和显著性
            for metric in ["accuracy", "recall", "precision", "roc_auc", "mcc", "avg_precision"]:
                z_scores = scipy.stats.zscore(df[metric])
                p_values = scipy.stats.norm.sf(abs(z_scores)) * 2
                significances = p_values <= pvalue_threshold
                df[f"{metric}_zscore"] = z_scores
                df[f"{metric}_pvalue"] = p_values
                df[f"{metric}_significance"] = significances
            return df
        else:
            flat_results = []
            for metric, values in result.items():
                flat_results.extend(values)
            z_scores = scipy.stats.zscore(flat_results)
            p_values = scipy.stats.norm.sf(abs(z_scores)) * 2
            significances = p_values <= pvalue_threshold
            return flat_results, z_scores, p_values, significances

    def __evaluate_model(self, X_train, Y_train, X_test, Y_test, scorer, progress_bar, cross_val=False, cv_kfolds=5):
        """根据提供的数据评估模型，并计算各项评价指标。"""
        self.base_model.fit(X_train, Y_train)
        metrics = {
            "accuracy": [],
            "recall": [],
            "precision": [],
            "roc_auc": [],
            "mcc": [],
            "avg_precision": []
        }

        # 首先计算原始模型的指标
        Y_pred = self.base_model.predict(X_test) if not cross_val else cross_val_predict(self.base_model, X_train, Y_train, cv=cv_kfolds)
        metrics["accuracy"].append(accuracy_score(Y_test, Y_pred))
        metrics["recall"].append(recall_score(Y_test, Y_pred, average='binary'))
        metrics["precision"].append(precision_score(Y_test, Y_pred, average='binary'))
        metrics["roc_auc"].append(roc_auc_score(Y_test, Y_pred))
        metrics["mcc"].append(matthews_corrcoef(Y_test, Y_pred))
        metrics["avg_precision"].append(average_precision_score(Y_test, Y_pred))

        # 然后计算扰乱模型的指标
        for _ in tqdm(range(self.iterations), disable=not progress_bar):
            Y_train_scrambled = np.random.permutation(Y_train)
            self.base_model.fit(X_train, Y_train_scrambled)
            Y_pred_scrambled = self.base_model.predict(X_test) if not cross_val else cross_val_predict(self.base_model, X_train, Y_train_scrambled, cv=cv_kfolds)
            metrics["accuracy"].append(accuracy_score(Y_test, Y_pred_scrambled))
            metrics["recall"].append(recall_score(Y_test, Y_pred_scrambled, average='binary'))
            metrics["precision"].append(precision_score(Y_test, Y_pred_scrambled, average='binary'))
            metrics["roc_auc"].append(roc_auc_score(Y_test, Y_pred_scrambled))
            metrics["mcc"].append(matthews_corrcoef(Y_test, Y_pred_scrambled))
            metrics["avg_precision"].append(average_precision_score(Y_test, Y_pred_scrambled))

        return metrics

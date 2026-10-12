
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pymongo import UpdateOne

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from config import (
    HOTEL_FEATURES_COLLECTION,
    CLUSTER_RESULTS_COLLECTION,
    TABLE_DIR,
    MODEL_DIR,
)
from mongo_utils import get_database


warnings.filterwarnings("ignore")


# CONFIG

MIN_REVIEWS_PER_HOTEL = 3
RANDOM_STATE = 42

# Cấu hình chốt sau thử nghiệm:
# no_text_no_group_no_topic + RobustScaler + PCA=8 + KMeans K=3
FINAL_K = 3
PCA_COMPONENTS_FOR_CLUSTERING = 8
PCA_COMPONENTS_FOR_VISUALIZATION = 2

K_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]

EXCLUDED_NUMERIC_FEATURES = {
    # Metadata / số lượng kỹ thuật, không dùng để phân cụm chất lượng khách sạn
    "total_rating",
    "num_reviews_in_file",
    "num_reviews_with_text_in_file",
    "num_reviews_imported_sample",

    # free_wifi bị thiếu nhiều, 0 có thể là missing chứ không phải điểm thật
    "free_wifi",

    # Cột sinh ra sau clustering cũ, không được dùng làm input
    "cluster",
    "final_cluster",
    "cluster_n_clusters",
    "pca_x",
    "pca_y",
}

TEXT_LENGTH_FEATURES = {
    "avg_char_count",
    "avg_word_count",
    "max_word_count",
    "short_review_ratio",
    "valid_embedding_ratio",
}

GROUP_TYPE_FEATURES = {
    "couple_ratio",
    "family_ratio",
    "solo_ratio",
    "group_ratio",
    "other_ratio",
    "unknown_ratio",
}



# LOAD DATA

def load_hotel_features():
    db = get_database()
    hotel_features_col = db[HOTEL_FEATURES_COLLECTION]

    data = list(hotel_features_col.find({}, {"_id": 0}))

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


# FEATURE SELECTION

def get_numeric_feature_columns(df):
    """
    Cấu hình feature cuối: no_text_no_group_no_topic.

    Giữ lại:
    - Rating features
    - Sentiment features
    - Evaluation categories
    - Aspect positive / negative / mention ratios
    - Country diversity features

    Loại bỏ:
    - topic_*: topic review chưa đủ rõ để đưa vào cụm cuối
    - text length: độ dài review không trực tiếp phản ánh chất lượng khách sạn
    - group type: loại nhóm khách không phải trọng tâm đề tài
    - metadata / cluster cũ
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = []

    for col in numeric_cols:
        if col in EXCLUDED_NUMERIC_FEATURES:
            continue

        if col.startswith("topic_"):
            continue

        if col in TEXT_LENGTH_FEATURES:
            continue

        if col in GROUP_TYPE_FEATURES:
            continue

        feature_cols.append(col)

    return feature_cols


def prepare_clustering_data(df):
    """
    Lọc khách sạn đủ dữ liệu, chọn feature, RobustScaler, PCA=8.
    """
    df = df.copy()

    if "num_reviews" not in df.columns:
        raise ValueError("Thiếu cột num_reviews trong hotel_features.")

    df = df[df["num_reviews"] >= MIN_REVIEWS_PER_HOTEL].copy()

    if df.empty:
        raise ValueError("Không còn khách sạn nào sau khi lọc num_reviews.")

    feature_cols = get_numeric_feature_columns(df)

    if not feature_cols:
        raise ValueError("Không tìm thấy numeric features để phân cụm.")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(
        PCA_COMPONENTS_FOR_CLUSTERING,
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    pca_for_clustering = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    X_cluster = pca_for_clustering.fit_transform(X_scaled)
    explained_variance = float(np.sum(pca_for_clustering.explained_variance_ratio_))

    return df, feature_cols, X_scaled, X_cluster, scaler, pca_for_clustering, explained_variance


# CLUSTERING METRICS

def evaluate_clustering(X, labels):
    labels = np.array(labels)

    non_noise_mask = labels != -1
    labels_non_noise = labels[non_noise_mask]
    X_non_noise = X[non_noise_mask]

    num_clusters = len(set(labels_non_noise))
    num_noise = int((labels == -1).sum())

    if num_clusters < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_index": None,
            "num_clusters": int(num_clusters),
            "num_noise": int(num_noise),
        }

    try:
        silhouette = silhouette_score(X_non_noise, labels_non_noise)
    except Exception:
        silhouette = None

    try:
        davies_bouldin = davies_bouldin_score(X_non_noise, labels_non_noise)
    except Exception:
        davies_bouldin = None

    try:
        calinski_harabasz = calinski_harabasz_score(X_non_noise, labels_non_noise)
    except Exception:
        calinski_harabasz = None

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": calinski_harabasz,
        "num_clusters": int(num_clusters),
        "num_noise": int(num_noise),
    }


# RUN ALGORITHMS

def run_kmeans(X):
    results = []

    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)

        results.append({
            "algorithm": "KMeans",
            "n_clusters": k,
            "inertia": float(model.inertia_),
            "labels": labels,
            "model": model,
            **metrics,
        })

    return results


def run_agglomerative(X):
    results = []

    for k in K_RANGE:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )

        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)

        results.append({
            "algorithm": "Agglomerative",
            "n_clusters": k,
            "labels": labels,
            "model": model,
            **metrics,
        })

    return results


def run_gmm(X):
    results = []

    for k in K_RANGE:
        model = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            covariance_type="full",
        )

        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)

        results.append({
            "algorithm": "GMM",
            "n_clusters": k,
            "labels": labels,
            "model": model,
            **metrics,
        })

    return results


def run_dbscan(X):
    results = []

    dbscan_configs = [
        {"eps": 0.5, "min_samples": 5},
        {"eps": 0.8, "min_samples": 5},
        {"eps": 1.0, "min_samples": 5},
        {"eps": 1.2, "min_samples": 5},
        {"eps": 1.5, "min_samples": 5},
        {"eps": 2.0, "min_samples": 5},
    ]

    for cfg in dbscan_configs:
        model = DBSCAN(
            eps=cfg["eps"],
            min_samples=cfg["min_samples"],
        )

        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)

        results.append({
            "algorithm": "DBSCAN",
            "n_clusters": metrics["num_clusters"],
            "eps": cfg["eps"],
            "min_samples": cfg["min_samples"],
            "labels": labels,
            "model": model,
            **metrics,
        })

    return results


def compare_clustering_algorithms(X):
    all_results = []

    print("===== RUN KMEANS =====")
    all_results.extend(run_kmeans(X))

    print("===== RUN AGGLOMERATIVE =====")
    all_results.extend(run_agglomerative(X))

    print("===== RUN GMM =====")
    all_results.extend(run_gmm(X))

    print("===== RUN DBSCAN =====")
    all_results.extend(run_dbscan(X))

    return all_results


# EXPORT METRICS

def build_metrics_dataframe(results):
    rows = []

    for item in results:
        rows.append({
            "algorithm": item["algorithm"],
            "n_clusters": item.get("n_clusters"),
            "eps": item.get("eps"),
            "min_samples": item.get("min_samples"),
            "inertia": item.get("inertia"),
            "silhouette_score": item.get("silhouette_score"),
            "davies_bouldin_index": item.get("davies_bouldin_index"),
            "calinski_harabasz_index": item.get("calinski_harabasz_index"),
            "num_noise": item.get("num_noise"),
        })

    metrics_df = pd.DataFrame(rows)

    valid_mask = (
        metrics_df["silhouette_score"].notna()
        & metrics_df["davies_bouldin_index"].notna()
        & metrics_df["calinski_harabasz_index"].notna()
    )

    metrics_df["selection_score"] = -999.0
    valid_df = metrics_df[valid_mask].copy()

    if not valid_df.empty:
        sil_min = valid_df["silhouette_score"].min()
        sil_max = valid_df["silhouette_score"].max()

        db_min = valid_df["davies_bouldin_index"].min()
        db_max = valid_df["davies_bouldin_index"].max()

        ch_min = valid_df["calinski_harabasz_index"].min()
        ch_max = valid_df["calinski_harabasz_index"].max()

        def normalize_high(value, min_value, max_value):
            if max_value == min_value:
                return 1.0
            return (value - min_value) / (max_value - min_value)

        def normalize_low(value, min_value, max_value):
            if max_value == min_value:
                return 1.0
            return (max_value - value) / (max_value - min_value)

        for idx, row in valid_df.iterrows():
            silhouette_norm = normalize_high(row["silhouette_score"], sil_min, sil_max)
            davies_bouldin_norm = normalize_low(row["davies_bouldin_index"], db_min, db_max)
            calinski_harabasz_norm = normalize_high(row["calinski_harabasz_index"], ch_min, ch_max)

            num_noise = row.get("num_noise", 0)
            if pd.isna(num_noise):
                num_noise = 0

            noise_penalty = 0.0
            if num_noise > 0:
                noise_penalty = min(1.0, num_noise / 500)

            selection_score = (
                0.45 * silhouette_norm
                + 0.30 * davies_bouldin_norm
                + 0.25 * calinski_harabasz_norm
                - noise_penalty
            )

            metrics_df.loc[idx, "selection_score"] = float(selection_score)

    metrics_df = metrics_df.sort_values("selection_score", ascending=False)

    return metrics_df


def select_final_result(results):
    """
    Chọn mô hình cuối theo định hướng đồ án: KMeans với K=3.
    """
    for item in results:
        if item["algorithm"] == "KMeans" and item.get("n_clusters") == FINAL_K:
            return item

    raise ValueError("Không tìm thấy kết quả KMeans với K=3.")


# PCA FOR VISUALIZATION

def create_pca_coordinates(X_cluster):
    pca = PCA(
        n_components=PCA_COMPONENTS_FOR_VISUALIZATION,
        random_state=RANDOM_STATE,
    )

    coords = pca.fit_transform(X_cluster)

    return coords, pca


# SAVE FINAL RESULT

def save_models(scaler, pca_for_clustering, pca_for_visualization, best_model, feature_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = MODEL_DIR / "hotel_feature_robust_scaler.pkl"
    pca_cluster_path = MODEL_DIR / "hotel_feature_pca_8.pkl"
    pca_visual_path = MODEL_DIR / "hotel_cluster_pca_2d.pkl"
    model_path = MODEL_DIR / "final_hotel_clustering_model.pkl"
    feature_cols_path = MODEL_DIR / "hotel_clustering_feature_columns.pkl"

    joblib.dump(scaler, scaler_path)
    joblib.dump(pca_for_clustering, pca_cluster_path)
    joblib.dump(pca_for_visualization, pca_visual_path)
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, feature_cols_path)

    print(f"Saved scaler: {scaler_path}")
    print(f"Saved PCA for clustering: {pca_cluster_path}")
    print(f"Saved PCA 2D: {pca_visual_path}")
    print(f"Saved final model: {model_path}")
    print(f"Saved feature columns: {feature_cols_path}")


def save_cluster_results_to_mongodb(df_result, best_result, pca_explained_variance):
    db = get_database()

    hotel_features_col = db[HOTEL_FEATURES_COLLECTION]
    cluster_results_col = db[CLUSTER_RESULTS_COLLECTION]

    algorithm = best_result["algorithm"]
    n_clusters = int(best_result.get("n_clusters", -1))

    operations_features = []
    operations_results = []
    now = datetime.utcnow()

    for _, row in df_result.iterrows():
        hotel_id = row["hotel_id"]

        cluster_doc = {
            "hotel_id": hotel_id,
            "hotel_name": row.get("hotel_name"),
            "province_id": row.get("province_id"),
            "province_name": row.get("province_name"),
            "cluster": int(row["cluster"]),
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            "scaler": "RobustScaler",
            "pca_components_for_clustering": PCA_COMPONENTS_FOR_CLUSTERING,
            "pca_explained_variance": float(pca_explained_variance),
            "pca_x": float(row["pca_x"]),
            "pca_y": float(row["pca_y"]),
            "num_reviews": int(row.get("num_reviews", 0)),
            "avg_review_rating": float(row.get("avg_review_rating", 0)),
            "positive_ratio": float(row.get("positive_ratio", 0)),
            "negative_ratio": float(row.get("negative_ratio", 0)),
            "avg_sentiment_score": float(row.get("avg_sentiment_score", 0)),
            "created_at": now,
            "updated_at": now,
        }

        operations_features.append(
            UpdateOne(
                {"hotel_id": hotel_id},
                {
                    "$set": {
                        "cluster": int(row["cluster"]),
                        "final_cluster": int(row["cluster"]),
                        "cluster_algorithm": algorithm,
                        "cluster_n_clusters": n_clusters,
                        "cluster_scaler": "RobustScaler",
                        "cluster_pca_components": PCA_COMPONENTS_FOR_CLUSTERING,
                        "cluster_pca_explained_variance": float(pca_explained_variance),
                        "pca_x": float(row["pca_x"]),
                        "pca_y": float(row["pca_y"]),
                        "cluster_updated_at": now,
                        "updated_at": now,
                    }
                },
            )
        )

        operations_results.append(
            UpdateOne(
                {"hotel_id": hotel_id, "algorithm": algorithm},
                {"$set": cluster_doc},
                upsert=True,
            )
        )

    if operations_features:
        hotel_features_col.bulk_write(operations_features, ordered=False)

    if operations_results:
        cluster_results_col.bulk_write(operations_results, ordered=False)

    print(f"Đã cập nhật cluster cho {len(df_result)} khách sạn vào MongoDB.")
    print(f"Đã lưu cluster_results cho {len(df_result)} khách sạn.")


def export_outputs(df_result, metrics_df, feature_cols, best_result, pca_explained_variance):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLE_DIR / "hotel_clustering_metrics.csv"
    df_result_path = TABLE_DIR / "hotel_cluster_results.csv"
    feature_cols_path = TABLE_DIR / "hotel_clustering_feature_columns.csv"
    cluster_summary_path = TABLE_DIR / "hotel_cluster_summary.csv"
    sample_path = TABLE_DIR / "sample_hotels_by_cluster.csv"
    config_path = TABLE_DIR / "hotel_clustering_final_config.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    df_result.to_csv(df_result_path, index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": feature_cols}).to_csv(feature_cols_path, index=False, encoding="utf-8-sig")

    config_df = pd.DataFrame([
        {
            "algorithm": best_result["algorithm"],
            "n_clusters": best_result.get("n_clusters"),
            "min_reviews_per_hotel": MIN_REVIEWS_PER_HOTEL,
            "feature_mode": "no_text_no_group_no_topic",
            "scaler": "RobustScaler",
            "pca_components_for_clustering": PCA_COMPONENTS_FOR_CLUSTERING,
            "pca_explained_variance": pca_explained_variance,
            "num_features_before_pca": len(feature_cols),
            "silhouette_score": best_result.get("silhouette_score"),
            "davies_bouldin_index": best_result.get("davies_bouldin_index"),
            "calinski_harabasz_index": best_result.get("calinski_harabasz_index"),
        }
    ])
    config_df.to_csv(config_path, index=False, encoding="utf-8-sig")

    cluster_summary = df_result.groupby("cluster").agg(
        num_hotels=("hotel_id", "count"),
        avg_review_rating=("avg_review_rating", "mean"),
        avg_hotel_rating=("hotel_rating", "mean"),
        positive_ratio=("positive_ratio", "mean"),
        neutral_ratio=("neutral_ratio", "mean"),
        negative_ratio=("negative_ratio", "mean"),
        avg_sentiment_score=("avg_sentiment_score", "mean"),
        avg_num_reviews=("num_reviews", "mean"),
        avg_country_diversity=("country_diversity", "mean"),
        avg_service_staff=("service_staff", "mean"),
        avg_cleanliness=("cleanliness", "mean"),
        avg_location=("location", "mean"),
        avg_value_for_money=("value_for_money", "mean"),
    ).reset_index()

    cluster_summary = cluster_summary.round(4)
    cluster_summary.to_csv(cluster_summary_path, index=False, encoding="utf-8-sig")

    samples = []
    for cluster in sorted(df_result["cluster"].unique()):
        cluster_df = df_result[df_result["cluster"] == cluster].head(20)
        samples.append(cluster_df)

    sample_df = pd.concat(samples, ignore_index=True)
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved cluster results: {df_result_path}")
    print(f"Saved feature columns: {feature_cols_path}")
    print(f"Saved cluster summary: {cluster_summary_path}")
    print(f"Saved sample hotels: {sample_path}")
    print(f"Saved final config: {config_path}")


# MAIN

def main():
    print("===== HOTEL CLUSTERING FINAL CONFIG =====")
    print("Feature mode: no_text_no_group_no_topic")
    print("Scaler: RobustScaler")
    print(f"PCA components for clustering: {PCA_COMPONENTS_FOR_CLUSTERING}")
    print(f"Final model: KMeans, K={FINAL_K}")

    df = load_hotel_features()

    if df.empty:
        print("Không có dữ liệu hotel_features.")
        return

    print(f"\nSố khách sạn ban đầu: {len(df)}")

    (
        df_valid,
        feature_cols,
        X_scaled,
        X_cluster,
        scaler,
        pca_for_clustering,
        pca_explained_variance,
    ) = prepare_clustering_data(df)

    print(f"Số khách sạn sau lọc num_reviews >= {MIN_REVIEWS_PER_HOTEL}: {len(df_valid)}")
    print(f"Số numeric features trước PCA: {len(feature_cols)}")
    print(f"PCA explained variance: {pca_explained_variance:.4f}")

    print("\n===== FEATURE COLUMNS USED =====")
    for i, col in enumerate(feature_cols, start=1):
        print(f"{i:02d}. {col}")

    results = compare_clustering_algorithms(X_cluster)
    metrics_df = build_metrics_dataframe(results)

    print("\n===== CLUSTERING METRICS =====")
    print(metrics_df)

    best_result = select_final_result(results)

    print("\n===== FINAL MODEL =====")
    print(f"Algorithm: {best_result['algorithm']}")
    print(f"n_clusters: {best_result.get('n_clusters')}")
    print(f"Silhouette: {best_result.get('silhouette_score')}")
    print(f"Davies-Bouldin: {best_result.get('davies_bouldin_index')}")
    print(f"Calinski-Harabasz: {best_result.get('calinski_harabasz_index')}")
    print(f"PCA explained variance: {pca_explained_variance}")

    labels = best_result["labels"]
    coords, pca_2d = create_pca_coordinates(X_cluster)

    df_result = df_valid.copy()
    df_result["cluster"] = labels.astype(int)
    df_result["pca_x"] = coords[:, 0]
    df_result["pca_y"] = coords[:, 1]

    save_models(
        scaler=scaler,
        pca_for_clustering=pca_for_clustering,
        pca_for_visualization=pca_2d,
        best_model=best_result["model"],
        feature_cols=feature_cols,
    )

    export_outputs(
        df_result=df_result,
        metrics_df=metrics_df,
        feature_cols=feature_cols,
        best_result=best_result,
        pca_explained_variance=pca_explained_variance,
    )

    save_cluster_results_to_mongodb(
        df_result=df_result,
        best_result=best_result,
        pca_explained_variance=pca_explained_variance,
    )

    print("\n===== FILE 10 FINISHED =====")


if __name__ == "__main__":
    main()

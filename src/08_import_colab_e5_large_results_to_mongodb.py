import pandas as pd
from datetime import datetime
from pymongo import UpdateOne

from config import REVIEWS_COLLECTION, TABLE_DIR
from mongo_utils import get_database


BATCH_SIZE = 500

E5_MODEL_NAME = "intfloat/multilingual-e5-large"
TOPIC_MODEL_NAME = "Multilingual-E5-Large + PCA + MiniBatchKMeans"


def import_colab_e5_large_results():
    db = get_database()
    reviews_col = db[REVIEWS_COLLECTION]

    input_path = TABLE_DIR / "colab_review_embeddings_topics_aspects_e5_large.pkl"

    if not input_path.exists():
        print(f"Không tìm thấy file: {input_path}")
        print("Hãy copy file colab_review_embeddings_topics_e5_large.pkl vào outputs/tables/")
        return

    df = pd.read_pickle(input_path)

    print("===== IMPORT COLAB E5-LARGE RESULTS TO MONGODB =====")
    print("Input:", input_path)
    print("Shape:", df.shape)

    required_cols = [
            "review_id",

            "embedding",
            "embedding_model",
            "embedding_dim",

            "review_topic",
            "topic_model",
            "topic_n_clusters",

            "positive_aspects",
            "negative_aspects",
            "review_aspects",
            "positive_aspect_scores",
            "negative_aspect_scores",
            "aspect_method",
            "aspect_model",
            "aspect_similarity_threshold",
        ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print("Thiếu cột trong file pkl:", missing_cols)
        return

    operations = []
    processed_count = 0

    for _, row in df.iterrows():
        review_id = row["review_id"]

        now = datetime.utcnow()

        update_doc = {
            "embedding": row["embedding"],
            "embedding_model": row["embedding_model"],
            "embedding_dim": int(row["embedding_dim"]),

            "review_topic": int(row["review_topic"]),
            "topic_model": row["topic_model"],
            "topic_n_clusters": int(row["topic_n_clusters"]),

            "positive_aspects": row["positive_aspects"],
            "negative_aspects": row["negative_aspects"],
            "review_aspects": row["review_aspects"],

            "positive_aspect_scores": row["positive_aspect_scores"],
            "negative_aspect_scores": row["negative_aspect_scores"],

            "aspect_method": row["aspect_method"],
            "aspect_model": row["aspect_model"],
            "aspect_similarity_threshold": float(row["aspect_similarity_threshold"]),

            "embedding_created_at": now,
            "topic_created_at": now,
            "aspect_updated_at": now,
            "updated_at": now,
        }

        operations.append(
            UpdateOne(
                {"review_id": review_id},
                {"$set": update_doc}
            )
        )

        processed_count += 1

        if len(operations) >= BATCH_SIZE:
            reviews_col.bulk_write(operations, ordered=False)
            operations = []

            print(f"Đã import: {processed_count}/{len(df)}")

    if operations:
        reviews_col.bulk_write(operations, ordered=False)

    print("\n===== IMPORT FINISHED =====")
    print(f"Số review trong file: {len(df)}")

    count_e5 = reviews_col.count_documents({
        "embedding_model": E5_MODEL_NAME
    })

    count_topic = reviews_col.count_documents({
        "topic_model": TOPIC_MODEL_NAME
    })

    count_with_embedding = reviews_col.count_documents({
        "embedding": {
            "$ne": None
        }
    })

    count_with_topic = reviews_col.count_documents({
        "review_topic": {
            "$ne": None
        }
    })

    print(f"Số review có E5-Large embedding: {count_e5}")
    print(f"Số review có E5-Large topic: {count_topic}")
    print(f"Tổng số review có embedding khác null: {count_with_embedding}")
    print(f"Tổng số review có review_topic khác null: {count_with_topic}")


if __name__ == "__main__":
    import_colab_e5_large_results()
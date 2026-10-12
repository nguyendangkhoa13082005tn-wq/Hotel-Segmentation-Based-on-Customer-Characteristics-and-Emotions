import json
import random
import hashlib
from datetime import datetime
from collections import defaultdict

from pymongo import UpdateOne

from config import (
    RAW_DATA_DIR,
    PROVINCES_COLLECTION,
    HOTELS_COLLECTION,
    REVIEWS_COLLECTION
)
from mongo_utils import get_database



# SAMPLE CONFIG

RANDOM_SEED = 42

# Mỗi tỉnh lấy tối đa bao nhiêu khách sạn.
# Tỉnh nào ít hơn số này thì lấy hết.
MAX_HOTELS_PER_PROVINCE = 50

# Mỗi khách sạn lấy tối đa bao nhiêu review có text.
MAX_REVIEWS_PER_HOTEL = 10

# Chia review theo điểm trong mỗi khách sạn.
POSITIVE_REVIEWS_PER_HOTEL = 4      
NEUTRAL_REVIEWS_PER_HOTEL = 3       
NEGATIVE_REVIEWS_PER_HOTEL = 3      

# Chia khách sạn theo điểm trong mỗi tỉnh.
# Tổng 20 + 15 + 15 = 50 khách sạn/tỉnh.
HIGH_RATING_HOTELS_PER_PROVINCE = 20      # hotel_rating >= 8
MID_RATING_HOTELS_PER_PROVINCE = 15       # 6 <= hotel_rating < 8
LOW_RATING_HOTELS_PER_PROVINCE = 15       # hotel_rating < 6

BATCH_SIZE_HOTELS = 1000
BATCH_SIZE_REVIEWS = 5000


# SAFE FUNCTIONS

def safe_string(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in ["", "nan", "none", "null"]:
        return None

    return text


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def safe_value(value):
    """
    Đảm bảo dữ liệu có thể lưu vào MongoDB.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [safe_value(v) for v in value]

    return value


def make_id(*parts):
    """
    Tạo ID ổn định để chạy lại không bị trùng.
    """
    raw = "_".join([str(p) for p in parts if p is not None])
    raw = raw.strip().lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# REVIEW HELPERS

def build_review_text(comment_positive, comment_negative):

    parts = []

    if comment_positive:
        parts.append(str(comment_positive).strip())

    if comment_negative:
        parts.append(str(comment_negative).strip())

    text = " ".join(parts).strip()

    if text == "":
        return None

    return text


def get_review_part(review_item):
    if not isinstance(review_item, dict):
        return {}

    review = review_item.get("review", {})

    if not isinstance(review, dict):
        return {}

    return review


def get_reviewer_part(review_item):
    if not isinstance(review_item, dict):
        return {}

    reviewer = review_item.get("reviewer", {})

    if not isinstance(reviewer, dict):
        return {}

    return reviewer


def has_review_text(review_item):
    review = get_review_part(review_item)

    positive = safe_string(review.get("comment_positive"))
    negative = safe_string(review.get("comment_negative"))

    return bool(positive or negative)


def get_review_score(review_item):
    review = get_review_part(review_item)
    return safe_float(review.get("score"))


def get_review_text_length(review_item):
    review = get_review_part(review_item)

    positive = safe_string(review.get("comment_positive"))
    negative = safe_string(review.get("comment_negative"))

    text = build_review_text(positive, negative)

    if not text:
        return 0

    return len(text)


def select_representative_reviews(reviews):
    """
    Chọn review đại diện trong 1 khách sạn:
    - Chỉ lấy review có text.
    - Chia theo score: positive / neutral / negative.
    - Ưu tiên review dài hơn vì có nhiều thông tin hơn cho embedding.
    """

    if not isinstance(reviews, list):
        return []

    reviews_with_text = [
        item for item in reviews
        if isinstance(item, dict) and has_review_text(item)
    ]

    positive = []
    neutral = []
    negative = []

    for item in reviews_with_text:
        score = get_review_score(item)

        if score is None:
            neutral.append(item)
        elif score >= 8:
            positive.append(item)
        elif score >= 6:
            neutral.append(item)
        else:
            negative.append(item)

    # Ưu tiên review dài hơn
    positive = sorted(positive, key=get_review_text_length, reverse=True)
    neutral = sorted(neutral, key=get_review_text_length, reverse=True)
    negative = sorted(negative, key=get_review_text_length, reverse=True)

    selected = []

    selected.extend(positive[:POSITIVE_REVIEWS_PER_HOTEL])
    selected.extend(neutral[:NEUTRAL_REVIEWS_PER_HOTEL])
    selected.extend(negative[:NEGATIVE_REVIEWS_PER_HOTEL])

    # Nếu chưa đủ MAX_REVIEWS_PER_HOTEL thì lấy thêm từ phần còn lại
    if len(selected) < MAX_REVIEWS_PER_HOTEL:
        selected_ids = set(id(x) for x in selected)

        remaining = [
            item for item in reviews_with_text
            if id(item) not in selected_ids
        ]

        remaining = sorted(remaining, key=get_review_text_length, reverse=True)

        needed = MAX_REVIEWS_PER_HOTEL - len(selected)
        selected.extend(remaining[:needed])

    return selected[:MAX_REVIEWS_PER_HOTEL]


# HOTEL HELPERS

def count_reviews_with_text(hotel_data):
    reviews = hotel_data.get("reviews", [])

    if not isinstance(reviews, list):
        return 0

    return sum(1 for item in reviews if has_review_text(item))


def get_hotel_rating(hotel_data):
    return safe_float(hotel_data.get("rating"))


def classify_hotel_rating(hotel_rating):
    if hotel_rating is None:
        return "mid"

    if hotel_rating >= 8:
        return "high"

    if hotel_rating >= 6:
        return "mid"

    return "low"


def build_hotel_info(file_path):
    """
    Đọc nhanh thông tin cần thiết để lấy mẫu khách sạn.
    """
    hotel_data = load_json_file(file_path)

    if not isinstance(hotel_data, dict):
        return None

    hotel_name = safe_string(hotel_data.get("name"))

    if hotel_name is None:
        hotel_name = file_path.stem.replace("_", " ").title()

    hotel_rating = get_hotel_rating(hotel_data)
    rating_group = classify_hotel_rating(hotel_rating)
    num_reviews_with_text = count_reviews_with_text(hotel_data)

    reviews = hotel_data.get("reviews", [])
    num_reviews_total = len(reviews) if isinstance(reviews, list) else 0

    return {
        "file_path": file_path,
        "hotel_name": hotel_name,
        "hotel_rating": hotel_rating,
        "rating_group": rating_group,
        "num_reviews_total": num_reviews_total,
        "num_reviews_with_text": num_reviews_with_text,
    }


def group_json_files_by_province():
    json_files = [
        file for file in RAW_DATA_DIR.rglob("*.json")
        if file.is_file()
    ]

    province_files = defaultdict(list)

    for file_path in json_files:
        province_id = file_path.parent.name
        province_files[province_id].append(file_path)

    return province_files


def select_hotels_for_one_province(province_id, files):
    """
    Chọn khách sạn đại diện cho một tỉnh:
    - Nếu tỉnh có <= MAX_HOTELS_PER_PROVINCE thì lấy hết.
    - Nếu nhiều hơn thì chia theo hotel_rating: high / mid / low.
    - Ưu tiên khách sạn có nhiều review có text.
    """

    hotel_infos = []

    for file_path in files:
        try:
            info = build_hotel_info(file_path)

            if info is not None:
                hotel_infos.append(info)

        except Exception:
            continue

    # Tỉnh ít khách sạn thì lấy hết
    if len(hotel_infos) <= MAX_HOTELS_PER_PROVINCE:
        return hotel_infos

    high = [h for h in hotel_infos if h["rating_group"] == "high"]
    mid = [h for h in hotel_infos if h["rating_group"] == "mid"]
    low = [h for h in hotel_infos if h["rating_group"] == "low"]

    # Ưu tiên khách sạn có nhiều review có text
    high = sorted(high, key=lambda x: x["num_reviews_with_text"], reverse=True)
    mid = sorted(mid, key=lambda x: x["num_reviews_with_text"], reverse=True)
    low = sorted(low, key=lambda x: x["num_reviews_with_text"], reverse=True)

    selected = []

    selected.extend(high[:HIGH_RATING_HOTELS_PER_PROVINCE])
    selected.extend(mid[:MID_RATING_HOTELS_PER_PROVINCE])
    selected.extend(low[:LOW_RATING_HOTELS_PER_PROVINCE])

    # Nếu thiếu do một nhóm không đủ, lấy thêm từ phần còn lại
    if len(selected) < MAX_HOTELS_PER_PROVINCE:
        selected_paths = set(str(x["file_path"]) for x in selected)

        remaining = [
            h for h in hotel_infos
            if str(h["file_path"]) not in selected_paths
        ]

        remaining = sorted(
            remaining,
            key=lambda x: x["num_reviews_with_text"],
            reverse=True
        )

        needed = MAX_HOTELS_PER_PROVINCE - len(selected)
        selected.extend(remaining[:needed])

    return selected[:MAX_HOTELS_PER_PROVINCE]


def sample_hotels_by_province(province_files):
    random.seed(RANDOM_SEED)

    sampled_hotel_infos = []

    for province_id, files in sorted(province_files.items()):
        selected = select_hotels_for_one_province(province_id, files)
        sampled_hotel_infos.extend(selected)

        print(
            f"Tỉnh {province_id}: tổng file={len(files)} | "
            f"chọn mẫu={len(selected)}"
        )

    return sampled_hotel_infos


# =========================================================
# IMPORT MAIN
# =========================================================

def import_sample_reviews():
    db = get_database()

    provinces_col = db[PROVINCES_COLLECTION]
    hotels_col = db[HOTELS_COLLECTION]
    reviews_col = db[REVIEWS_COLLECTION]

    province_ops = []
    hotel_ops = []
    review_ops = []

    total_files_success = 0
    total_files_error = 0

    total_hotels_imported = 0
    total_reviews_imported = 0

    total_positive_reviews = 0
    total_neutral_reviews = 0
    total_negative_reviews = 0

    if not RAW_DATA_DIR.exists():
        print(f"Không tìm thấy thư mục raw data: {RAW_DATA_DIR}")
        return

    province_files = group_json_files_by_province()
    sampled_hotel_infos = sample_hotels_by_province(province_files)

    total_original_files = sum(len(v) for v in province_files.values())

    print("\n===== IMPORT SAMPLE HOTEL REVIEWS TO MONGODB =====")
    print(f"Số tỉnh/thành tìm thấy: {len(province_files)}")
    print(f"Tổng số file JSON gốc: {total_original_files}")
    print(f"Số khách sạn mẫu sẽ import: {len(sampled_hotel_infos)}")
    print(f"Mỗi tỉnh lấy tối đa: {MAX_HOTELS_PER_PROVINCE} khách sạn")
    print(f"Mỗi khách sạn lấy tối đa: {MAX_REVIEWS_PER_HOTEL} review có text")
    print(f"Raw data dir: {RAW_DATA_DIR}")

    for file_index, hotel_info in enumerate(sampled_hotel_infos, start=1):
        file_path = hotel_info["file_path"]

        try:
            province_id = file_path.parent.name
            province_name = province_id.replace("-", " ").title()

            hotel_data = load_json_file(file_path)

            if not isinstance(hotel_data, dict):
                total_files_error += 1
                continue

            hotel_name = safe_string(hotel_data.get("name"))

            if hotel_name is None:
                hotel_name = file_path.stem.replace("_", " ").title()

            hotel_id = make_id(province_id, hotel_name, str(file_path))

            address = safe_string(hotel_data.get("address"))
            description = safe_string(hotel_data.get("description"))
            hotel_rating = safe_float(hotel_data.get("rating"))
            total_rating = safe_int(hotel_data.get("total_rating"))
            evaluation_categories = safe_value(
                hotel_data.get("evaluation_categories", {})
            )

            all_reviews = hotel_data.get("reviews", [])

            if not isinstance(all_reviews, list):
                all_reviews = []

            selected_reviews = select_representative_reviews(all_reviews)

            province_ops.append(
                UpdateOne(
                    {"province_id": province_id},
                    {
                        "$set": {
                            "province_id": province_id,
                            "province_name": province_name,
                            "source_folder": str(file_path.parent),
                            "updated_at": datetime.utcnow()
                        },
                        "$setOnInsert": {
                            "created_at": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
            )

            raw_hotel_data = dict(hotel_data)
            raw_hotel_data.pop("reviews", None)

            hotel_ops.append(
                UpdateOne(
                    {"hotel_id": hotel_id},
                    {
                        "$set": {
                            "hotel_id": hotel_id,
                            "hotel_name": hotel_name,
                            "province_id": province_id,
                            "province_name": province_name,

                            "address": address,
                            "description": description,
                            "hotel_rating": hotel_rating,
                            "hotel_rating_group": classify_hotel_rating(hotel_rating),
                            "total_rating": total_rating,
                            "evaluation_categories": evaluation_categories,

                            "num_reviews_in_file": len(all_reviews),
                            "num_reviews_with_text_in_file": count_reviews_with_text(hotel_data),
                            "num_reviews_imported_sample": len(selected_reviews),

                            "source_file": str(file_path),
                            "raw_hotel_data": safe_value(raw_hotel_data),

                            "is_sample_hotel": True,
                            "sample_rule": {
                                "max_hotels_per_province": MAX_HOTELS_PER_PROVINCE,
                                "max_reviews_per_hotel": MAX_REVIEWS_PER_HOTEL,
                                "positive_reviews_per_hotel": POSITIVE_REVIEWS_PER_HOTEL,
                                "neutral_reviews_per_hotel": NEUTRAL_REVIEWS_PER_HOTEL,
                                "negative_reviews_per_hotel": NEGATIVE_REVIEWS_PER_HOTEL,
                            },

                            "updated_at": datetime.utcnow()
                        },
                        "$setOnInsert": {
                            "created_at": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
            )

            total_hotels_imported += 1

            for review_index, item in enumerate(selected_reviews):
                reviewer = get_reviewer_part(item)
                review = get_review_part(item)

                reviewer_name = safe_string(reviewer.get("name"))
                reviewer_avatar = safe_string(reviewer.get("avatar"))
                customer_country = safe_string(reviewer.get("country"))

                review_date = safe_string(review.get("date"))
                review_title = safe_string(review.get("rating"))
                review_score = safe_float(review.get("score"))
                room_type = safe_string(review.get("room_type"))
                stay_duration = safe_string(review.get("stay_duration"))
                group_type = safe_string(review.get("group_type"))

                positive_review = safe_string(review.get("comment_positive"))
                negative_review = safe_string(review.get("comment_negative"))

                review_text = build_review_text(
                    positive_review,
                    negative_review
                )

                # Bo review khong co noi dung text
                # Vi review nay khong dung duoc cho embedding / NLP
                if review_text is None:
                    continue

                if review_score is None:
                    review_rating_group = "neutral"
                    total_neutral_reviews += 1
                elif review_score >= 8:
                    review_rating_group = "positive"
                    total_positive_reviews += 1
                elif review_score >= 6:
                    review_rating_group = "neutral"
                    total_neutral_reviews += 1
                else:
                    review_rating_group = "negative"
                    total_negative_reviews += 1

                review_id = make_id(
                    province_id,
                    hotel_name,
                    reviewer_name,
                    customer_country,
                    review_date,
                    review_score,
                    positive_review,
                    negative_review,
                    review_index,
                    str(file_path)
                )

                review_ops.append(
                    UpdateOne(
                        {"review_id": review_id},
                        {
                            "$set": {
                                "review_id": review_id,
                                "hotel_id": hotel_id,
                                "hotel_name": hotel_name,
                                "province_id": province_id,
                                "province_name": province_name,

                                "reviewer_name": reviewer_name,
                                "reviewer_avatar": reviewer_avatar,
                                "customer_country": customer_country,

                                "review_date": review_date,
                                "review_title": review_title,
                                "rating": review_score,
                                "review_rating_group": review_rating_group,
                                "room_type": room_type,
                                "stay_duration": stay_duration,
                                "group_type": group_type,

                                "positive_review": positive_review,
                                "negative_review": negative_review,
                                "review_text": review_text,

                                "raw_review_data": safe_value(item),

                                "is_sample_review": True,
                                "is_embedding_sample": review_text is not None,

                                "language": None,
                                "cleaned_text": None,
                                "translated_text": None,

                                "sentiment_label": None,
                                "sentiment_score": None,

                                "embedding": None,
                                "review_topic": None,
                                "review_aspects": [],

                                "source_file": str(file_path),
                                "source_review_index": review_index,
                                "processed_status": "raw",

                                "updated_at": datetime.utcnow()
                            },
                            "$setOnInsert": {
                                "created_at": datetime.utcnow()
                            }
                        },
                        upsert=True
                    )
                )

                total_reviews_imported += 1

                if len(review_ops) >= BATCH_SIZE_REVIEWS:
                    reviews_col.bulk_write(review_ops, ordered=False)
                    review_ops = []

            if len(hotel_ops) >= BATCH_SIZE_HOTELS:
                hotels_col.bulk_write(hotel_ops, ordered=False)
                hotel_ops = []

            if len(province_ops) >= 100:
                provinces_col.bulk_write(province_ops, ordered=False)
                province_ops = []

            total_files_success += 1

            if file_index % 500 == 0:
                print(
                    f"Đã xử lý {file_index}/{len(sampled_hotel_infos)} khách sạn mẫu | "
                    f"hotels={total_hotels_imported} | "
                    f"reviews={total_reviews_imported}"
                )

        except Exception as e:
            total_files_error += 1
            print(f"\nLỗi file: {file_path}")
            print(f"Lý do: {e}")

    if province_ops:
        provinces_col.bulk_write(province_ops, ordered=False)

    if hotel_ops:
        hotels_col.bulk_write(hotel_ops, ordered=False)

    if review_ops:
        reviews_col.bulk_write(review_ops, ordered=False)

    print("\n===== SAMPLE IMPORT FINISHED =====")
    print(f"Số tỉnh/thành: {len(province_files)}")
    print(f"Tổng số file JSON gốc: {total_original_files}")
    print(f"Số khách sạn mẫu đã chọn: {len(sampled_hotel_infos)}")
    print(f"Số file xử lý thành công: {total_files_success}")
    print(f"Số file lỗi: {total_files_error}")
    print(f"Số khách sạn mẫu đã import: {total_hotels_imported}")
    print(f"Số review mẫu đã import: {total_reviews_imported}")
    print(f"Review positive: {total_positive_reviews}")
    print(f"Review neutral: {total_neutral_reviews}")
    print(f"Review negative: {total_negative_reviews}")
    print("Database: hotel_review_clustering")
    print("Collections: provinces, hotels, reviews")


if __name__ == "__main__":
    import_sample_reviews()

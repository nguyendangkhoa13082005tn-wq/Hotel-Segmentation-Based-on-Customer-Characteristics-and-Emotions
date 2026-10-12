from config import (
    PROVINCES_COLLECTION,
    HOTELS_COLLECTION,
    REVIEWS_COLLECTION,
    HOTEL_FEATURES_COLLECTION,
    CLUSTER_RESULTS_COLLECTION
)
from mongo_utils import get_database


def create_indexes(db):
    """
    Tạo index cho các collection chính.
    Index giúp MongoDB tìm kiếm nhanh hơn.
    """

    provinces = db[PROVINCES_COLLECTION]
    hotels = db[HOTELS_COLLECTION]
    reviews = db[REVIEWS_COLLECTION]
    hotel_features = db[HOTEL_FEATURES_COLLECTION]
    cluster_results = db[CLUSTER_RESULTS_COLLECTION]

    # Collection provinces
    provinces.create_index("province_id", unique=True)

    # Collection hotels
    hotels.create_index("hotel_id", unique=True)
    hotels.create_index("province_id")
    hotels.create_index("hotel_name")
    hotels.create_index("hotel_rating")
    hotels.create_index("hotel_rating_group")
    hotels.create_index("is_sample_hotel")

    # Collection reviews
    reviews.create_index("review_id", unique=True)
    reviews.create_index("hotel_id")
    reviews.create_index("province_id")
    reviews.create_index("customer_country")
    reviews.create_index("rating")
    reviews.create_index("review_rating_group")
    reviews.create_index("language")
    reviews.create_index("sentiment_label")
    reviews.create_index("review_topic")
    reviews.create_index("review_aspects")
    reviews.create_index("processed_status")
    reviews.create_index("is_sample_review")
    reviews.create_index("is_embedding_sample")

    # Collection dùng sau này
    hotel_features.create_index("hotel_id", unique=True)
    hotel_features.create_index("province_id")
    hotel_features.create_index("cluster")

    cluster_results.create_index("hotel_id")
    cluster_results.create_index("cluster")
    cluster_results.create_index("algorithm")

    print("Đã tạo index MongoDB thành công.")


def check_basic_counts(db):
    """
    Kiểm tra số lượng dữ liệu chính sau khi import.
    """

    provinces = db[PROVINCES_COLLECTION]
    hotels = db[HOTELS_COLLECTION]
    reviews = db[REVIEWS_COLLECTION]

    print("\n===== MONGODB DATA SUMMARY =====")
    print("Số tỉnh/thành:", provinces.count_documents({}))
    print("Số khách sạn mẫu:", hotels.count_documents({}))
    print("Số review mẫu:", reviews.count_documents({}))

    print("Số review có text:", reviews.count_documents({
        "review_text": {
            "$nin": [None, ""]
        }
    }))

    print("Số review không có text:", reviews.count_documents({
        "review_text": {
            "$in": [None, ""]
        }
    }))

    print("Số review dùng để embedding:", reviews.count_documents({
        "is_embedding_sample": True
    }))


def check_review_groups(db):
    """
    Kiểm tra mẫu có đủ review tốt / trung bình / xấu không.
    """

    reviews = db[REVIEWS_COLLECTION]

    print("\n===== REVIEW RATING GROUP SUMMARY =====")
    print("Review positive:", reviews.count_documents({
        "review_rating_group": "positive"
    }))
    print("Review neutral:", reviews.count_documents({
        "review_rating_group": "neutral"
    }))
    print("Review negative:", reviews.count_documents({
        "review_rating_group": "negative"
    }))


def check_hotel_groups(db):
    """
    Kiểm tra mẫu có đủ khách sạn rating cao / trung bình / thấp không.
    """

    hotels = db[HOTELS_COLLECTION]

    print("\n===== HOTEL RATING GROUP SUMMARY =====")
    print("Hotel high rating:", hotels.count_documents({
        "hotel_rating_group": "high"
    }))
    print("Hotel mid rating:", hotels.count_documents({
        "hotel_rating_group": "mid"
    }))
    print("Hotel low rating:", hotels.count_documents({
        "hotel_rating_group": "low"
    }))


def check_reviews_by_province(db):
    """
    Kiểm tra mỗi tỉnh có bao nhiêu khách sạn và bao nhiêu review mẫu.
    """

    reviews = db[REVIEWS_COLLECTION]

    print("\n===== SAMPLE REVIEWS BY PROVINCE =====")

    pipeline = [
        {
            "$group": {
                "_id": "$province_id",
                "num_reviews": {"$sum": 1},
                "hotel_ids": {"$addToSet": "$hotel_id"},
                "countries": {"$addToSet": "$customer_country"},
                "positive_reviews": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$review_rating_group", "positive"]},
                            1,
                            0
                        ]
                    }
                },
                "neutral_reviews": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$review_rating_group", "neutral"]},
                            1,
                            0
                        ]
                    }
                },
                "negative_reviews": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$review_rating_group", "negative"]},
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "province_id": "$_id",
                "num_reviews": 1,
                "num_hotels": {"$size": "$hotel_ids"},
                "num_customer_countries": {"$size": "$countries"},
                "positive_reviews": 1,
                "neutral_reviews": 1,
                "negative_reviews": 1
            }
        },
        {
            "$sort": {
                "province_id": 1
            }
        }
    ]

    for row in reviews.aggregate(pipeline):
        print(row)


def check_hotels_by_province(db):
    """
    Kiểm tra số khách sạn mẫu theo từng tỉnh.
    """

    hotels = db[HOTELS_COLLECTION]

    print("\n===== SAMPLE HOTELS BY PROVINCE =====")

    pipeline = [
        {
            "$group": {
                "_id": "$province_id",
                "num_hotels": {"$sum": 1},
                "high_hotels": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$hotel_rating_group", "high"]},
                            1,
                            0
                        ]
                    }
                },
                "mid_hotels": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$hotel_rating_group", "mid"]},
                            1,
                            0
                        ]
                    }
                },
                "low_hotels": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$hotel_rating_group", "low"]},
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "province_id": "$_id",
                "num_hotels": 1,
                "high_hotels": 1,
                "mid_hotels": 1,
                "low_hotels": 1
            }
        },
        {
            "$sort": {
                "province_id": 1
            }
        }
    ]

    for row in hotels.aggregate(pipeline):
        print(row)


def show_sample_documents(db):
    """
    In thử 1 khách sạn và 1 review để kiểm tra field đã đúng chưa.
    """

    hotels = db[HOTELS_COLLECTION]
    reviews = db[REVIEWS_COLLECTION]

    print("\n===== SAMPLE HOTEL DOCUMENT =====")
    sample_hotel = hotels.find_one({}, {"_id": 0})

    if sample_hotel:
        for key, value in sample_hotel.items():
            if key == "raw_hotel_data":
                print(f"{key}: <raw hotel data>")
            elif key == "description":
                text = str(value)
                print(f"{key}: {text[:200]}...")
            else:
                print(f"{key}: {value}")
    else:
        print("Chưa có hotel nào.")

    print("\n===== SAMPLE REVIEW DOCUMENT =====")
    sample_review = reviews.find_one({}, {"_id": 0})

    if sample_review:
        for key, value in sample_review.items():
            if key == "raw_review_data":
                print(f"{key}: <raw review data>")
            elif key == "review_text":
                text = str(value)
                print(f"{key}: {text[:300]}...")
            else:
                print(f"{key}: {value}")
    else:
        print("Chưa có review nào.")


def main():
    db = get_database()

    print("===== SETUP AND CHECK MONGODB =====")

    create_indexes(db)

    check_basic_counts(db)
    check_hotel_groups(db)
    check_review_groups(db)
    check_hotels_by_province(db)
    check_reviews_by_province(db)
    show_sample_documents(db)

    print("\nHoàn tất setup và kiểm tra MongoDB.")


if __name__ == "__main__":
    main()
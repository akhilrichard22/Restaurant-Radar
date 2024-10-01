# Databricks notebook source
# MAGIC %pip install requests
# MAGIC %pip install pyodbc
# MAGIC %pip install sqlalchemy
# MAGIC %pip install pyspark

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import requests
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("YelpDataLoader").getOrCreate()

url = "https://api.yelp.com/v3/businesses/search?location=New%20York&term=restaurants&radius=40000&categories=&sort_by=best_match&limit=50"
url = 'https://api.yelp.com/v3/businesses/search?location=New%20York&term=restaurants&radius=40000&categories=&sort_by=best_match&limit=50'

headers = {
    "Authorization": "Bearer 9d0Mdqt-HFT6uxfxre407sR4dckCNROifIWaGHqK1ojSlEznlQF35GOuYbQg_sJsSWrWIYFYxiXCMFiu2NuzUtUeZdtmAdAU3CX62CRZqkzb-CKbKxt_5RGVVJuOZnYx",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

# Ensure the request was successful
if response.status_code == 200:
    json_data = response.json()
else:
    print(f"Error: Unable to fetch data. Status code: {response.status_code}")
    json_data = {}

# Function to parse JSON data
def parse_json(data):
    businesses = []
    for business in data.get('businesses', []):
        business_id = business.get('id', 'N/A')
        name = business.get('name', 'N/A')
        alias = business.get('alias', 'N/A')
        image_url = business.get('image_url', 'N/A')
        is_closed = business.get('is_closed', 'N/A')
        url = business.get('url', 'N/A')
        review_count = business.get('review_count', 'N/A')
        rating = business.get('rating', 'N/A')
        categories = [category['title'] for category in business.get('categories', [])]
        latitude = business['coordinates'].get('latitude', 'N/A')
        longitude = business['coordinates'].get('longitude', 'N/A')
        transactions = business.get('transactions', [])
        price = business.get('price', 'N/A')
        address = ", ".join(business['location'].get('display_address', []))
        phone = business.get('phone', 'N/A')
        distance = business.get('distance', 'N/A')
        business_hours = business.get('business_hours', [])
        attributes = business.get('attributes', {})

        businesses.append({
            "business_id": business_id,
            "name": name,
            "alias": alias,
            "image_url": image_url,
            "is_closed": is_closed,
            "url": url,
            "review_count": review_count,
            "rating": rating,
            "categories": categories,
            "latitude": latitude,
            "longitude": longitude,
            "transactions": transactions,
            "price": price,
            "address": address,
            "phone": phone,
            "distance": distance,
            "business_hours": business_hours,
            "attributes": attributes
        })
    return businesses

# Parse the JSON data if available
if json_data:
    print(json_data)
    businesses_data = parse_json(json_data)
else:
    businesses_data = []

# Create a Spark DataFrame from the parsed data
if businesses_data:
    businesses_df = spark.createDataFrame(businesses_data)
    display(businesses_df)
else:
    print("No data to load into DataFrame.")


# COMMAND ----------

# import requests
# from pyspark.sql import SparkSession

# # Define JDBC connection properties
# jdbc_url = "jdbc:sqlserver://restaurant-radar-server.database.windows.net:1433;database=Restaurant_Radar_Database;"
# db_properties = {
#     "user": "akvincen",
#     "password": "Mar!nes007",
#     "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
# }

# # Function to insert business data into the database
# def insert_business_data(row):
#     from pyspark.sql import SparkSession
#     spark = SparkSession.builder.getOrCreate()

#     # Extract relevant fields from the row
#     business_id = row['id']
#     name = row['name']
#     alias = row.get('alias', 'N/A')
#     image_url = row.get('image_url', 'N/A')
#     is_closed = row['is_closed']
#     url = row['url']
#     review_count = row['review_count']
#     rating = row['rating']
#     categories = ', '.join([category['title'] for category in row.get('categories', [])])
#     latitude = row['coordinates']['latitude']
#     longitude = row['coordinates']['longitude']
#     transactions = ', '.join(row.get('transactions', []))
#     price = row.get('price', 'N/A')
#     address = ', '.join(row['location']['display_address'])
#     phone = row.get('phone', 'N/A')
#     distance = row.get('distance', 'N/A')

    
#    # Create a DataFrame with the current business data
#     df_business = spark.createDataFrame([
#         (business_id, name, alias, image_url, is_closed, url, review_count, rating,
#         categories, latitude, longitude, transactions, price, address, phone, distance)
#     ], ["business_id", "name", "alias", "image_url", "is_closed", "url", "review_count", "rating",
#         "categories", "latitude", "longitude", "transactions", "price", "address", "phone", "distance"])


#     # Write DataFrame to SQL database using JDBC
#     try:
#         df_business.write \
#             .format("jdbc") \
#             .option("url", jdbc_url) \
#             .option("dbtable", "restaurants") \
#             .option("user", db_properties['user']) \
#             .option("password", db_properties['password']) \
#             .option("driver", db_properties['driver']) \
#             .mode("append") \
#             .save()
#         print(f"Inserted data for business ID {business_id}")
#     except Exception as e:
#         print(f"Error inserting data for business ID {business_id}: {str(e)}")

# # Main function to fetch data from Yelp and process
# def main():
#     spark = SparkSession.builder \
#         .appName("Yelp Business Data") \
#         .getOrCreate()

#     businesses = json_data.get('businesses', [])

#     # Iterate over businesses and insert into SQL database
#     for business in businesses:
#         insert_business_data(business)

# if __name__ == "__main__":
#     main()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import array_join, udf, col, lit, lower, trim, split, array_contains
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create a Spark session
spark = SparkSession.builder.appName("EnhancedRecommendationSystem").getOrCreate()

# Combine categories into a single string and normalize
businesses_df = businesses_df.withColumn("categories_str", lower(trim(array_join("categories", ", "))))

# Select relevant features
feature_columns = ["rating", "review_count"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
feature_df = assembler.transform(businesses_df)

# Normalize the feature vectors
normalizer = Normalizer(inputCol="features", outputCol="norm_features")
normalized_df = normalizer.transform(feature_df)

# Capture user preferences
user_categories_input = input("Enter preferred categories (comma separated): ")
user_categories = [cat.strip().lower() for cat in user_categories_input.split(",")]
user_rating = float(input("Enter minimum preferred rating: "))
user_review_count = float(input("Enter minimum review count: "))

# Create a vector for user preferences
user_vector = np.array([user_rating, user_review_count])
user_vector_norm = user_vector / np.linalg.norm(user_vector)  # Normalize user vector

# Function to calculate cosine similarity
def calculate_cosine_similarity(norm_features):
    restaurant_vector = np.array(norm_features)
    similarity = cosine_similarity([user_vector_norm], [restaurant_vector])
    return float(similarity[0][0])

# Register the UDF
cosine_similarity_udf = udf(calculate_cosine_similarity, DoubleType())

# Calculate similarity scores
similarity_df = normalized_df.withColumn("similarity", cosine_similarity_udf(col("norm_features")))

# Split categories_str into an array of categories
similarity_df = similarity_df.withColumn("categories_array", split(col("categories_str"), ", "))

# Filter based on categories, rating, and review count, and sort by similarity
if user_categories:
    for category in user_categories:
        similarity_df = similarity_df.filter(array_contains(col("categories_array"), category))
    recommendations = (similarity_df
                       .filter(col("rating") >= user_rating)
                       .filter(col("review_count") >= user_review_count)
                       .orderBy(col("similarity"), ascending=False)
                       .limit(10))
else:
    recommendations = (similarity_df
                       .filter(col("rating") >= user_rating)
                       .filter(col("review_count") >= user_review_count)
                       .orderBy(col("similarity"), ascending=False)
                       .limit(10))

# Show top recommendations
recommendations.select("business_id", "name", "categories_str", "rating", "review_count", "similarity").show()


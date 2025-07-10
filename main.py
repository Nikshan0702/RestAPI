
# from flask import Flask, jsonify  

# app = Flask(__name__)


# @app.route("/", methods=["GET"])
# def recommend(product_id):
#     try:
        
#         index = df[df["productId"] == product_id].index[0]  

#     except IndexError:
       
#         return jsonify({"error": "Product not found"}), 404

   
#     cosine_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
#     similar_indices = cosine_scores.argsort()[::-1][1:4]
#     recommended = df.iloc[similar_indices][["productId", "name", "brand", "color"]]


#     return jsonify(recommended.to_dict(orient="records"))


# if __name__ == "__main__":
#     app.run(debug=True)


# ✅ Flask - Import Flask framework
from flask import Flask, jsonify

# 🟨 pandas - For reading and manipulating tabular data
import pandas as pd

# 🟩 scikit-learn - For text vectorization and similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # 🟩
from sklearn.metrics.pairwise import cosine_similarity       # 🟩

# ✅ Flask - Initialize Flask app
app = Flask(__name__)

# 🟨 pandas - Read product data from JSON file
# 🟪 user-defined - 'products.json' is your custom dataset
df = pd.read_json("products.json")

# 🟪 user-defined - Combine fields to use for recommendation logic
# 🟨 pandas - Create new column using existing columns
df["combined"] = df["category"] + " " + df["brand"] + " " + df["color"]

# 🟩 scikit-learn - Create TF-IDF vectorizer object
tfidf = TfidfVectorizer()

# 🟩 scikit-learn - Convert the combined text into numerical vectors
tfidf_matrix = tfidf.fit_transform(df["combined"])

# ✅ Flask - Create route to recommend similar products
@app.route("/recommend/<product_id>", methods=["GET"])  # 🟪 user-defined route name
def recommend(product_id):  # 🟦 Python - Function definition
    try:
        # 🟨 pandas - Find index of the product that matches input product_id
        # 🟪 user-defined - 'productId' is your dataset field
        index = df[df["productId"] == product_id].index[0]
    except IndexError:
        # ✅ Flask - Return 404 if product not found
        return jsonify({"error": "Product not found"}), 404

    # 🟩 scikit-learn - Calculate cosine similarity for this product vs all
    cosine_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()

    # 🟪 user-defined - Get top 3 most similar products (excluding the original)
    similar_indices = cosine_scores.argsort()[::-1][1:4]

    # 🟨 pandas - Get details of the recommended products
    # 🟪 user-defined - Choose which fields to return
    recommended = df.iloc[similar_indices][["productId", "name", "brand", "color"]]

    # ✅ Flask - Convert to JSON and send response
    return jsonify(recommended.to_dict(orient="records"))

# ✅ Flask - Run the app in debug mode (development only)
if __name__ == "__main__":
    app.run(debug=True)


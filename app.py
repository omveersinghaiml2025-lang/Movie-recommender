
import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# 1. Configure the website
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Movie Recommendation System")
st.write(
    "Select a movie you like to discover similar movies "
    "using machine learning."
)


# 2. Locate the dataset files
BASE_DIR = Path(__file__).resolve().parent

MOVIES_FILE = BASE_DIR / "tmdb_5000_movies.csv"

CREDITS_PART1_FILE = BASE_DIR / "tmdb_5000_credits_part1.csv"
CREDITS_PART2_FILE = BASE_DIR / "tmdb_5000_credits_part2.csv"


# 3. Load and combine the CSV files
@st.cache_data
def load_data():
    # Check that all expected files exist
    required_files = [
        MOVIES_FILE,
        CREDITS_PART1_FILE,
        CREDITS_PART2_FILE
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path.name}"
            )

    movies = pd.read_csv(MOVIES_FILE)

    credits_part1 = pd.read_csv(CREDITS_PART1_FILE)
    credits_part2 = pd.read_csv(CREDITS_PART2_FILE)

    # Combine the two credits parts
    credits = pd.concat(
        [credits_part1, credits_part2],
        ignore_index=True
    )

    # Keep only the credit columns we need
    required_credit_columns = ["movie_id", "cast", "crew"]

    missing_columns = [
        column for column in required_credit_columns
        if column not in credits.columns
    ]

    if missing_columns:
        raise ValueError(
            "Credits CSV is missing columns: "
            + ", ".join(missing_columns)
        )

    credits = credits[required_credit_columns]

    # Match credits to movies by movie ID
    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    return movies


# 4. Convert genre, keyword and cast data into names
def extract_names(value, limit=None):
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        items = ast.literal_eval(value)

        if not isinstance(items, list):
            return []

        names = [
            item["name"].replace(" ", "")
            for item in items
            if isinstance(item, dict) and item.get("name")
        ]

        return names[:limit] if limit else names

    except (ValueError, SyntaxError, TypeError):
        return []


# 5. Find the movie director
def get_director(value):
    if not isinstance(value, str) or not value.strip():
        return ""

    try:
        crew = ast.literal_eval(value)

        for person in crew:
            if (
                isinstance(person, dict)
                and person.get("job") == "Director"
            ):
                return person.get("name", "").replace(" ", "")

    except (ValueError, SyntaxError, TypeError):
        pass

    return ""


# 6. Prepare the machine learning model
@st.cache_resource
def build_model():
    movies = load_data()

    required_columns = [
        "title", "genres", "keywords",
        "cast", "crew", "overview"
    ]

    for column in required_columns:
        if column not in movies.columns:
            movies[column] = ""

    movies = movies.dropna(subset=["title"]).copy()

    for column in [
        "genres", "keywords", "cast", "crew", "overview"
    ]:
        movies[column] = movies[column].fillna("")

    movies["genres_list"] = movies["genres"].apply(
        extract_names
    )

    movies["keywords_list"] = movies["keywords"].apply(
        extract_names
    )

    movies["cast_list"] = movies["cast"].apply(
        lambda value: extract_names(value, limit=3)
    )

    movies["director_name"] = movies["crew"].apply(
        get_director
    )

    # Combine important movie information
    def make_tags(row):
        parts = (
            row["genres_list"]
            + row["keywords_list"]
            + row["cast_list"]
            + [row["director_name"]]
            + [str(row["overview"])]
        )

        return " ".join(
            part for part in parts if part
        ).lower()

    movies["tags"] = movies.apply(make_tags, axis=1)

    # Convert movie text into numerical vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(movies["tags"])

    return movies.reset_index(drop=True), tfidf_matrix


# 7. Start the app and load the model
try:
    with st.spinner("Loading movies and building the AI model..."):
        movies, tfidf_matrix = build_model()

except FileNotFoundError as error:
    st.error(f"Dataset file missing: {error.filename or error}")
    st.info(
        "Ensure app.py and all three CSV files are in the "
        "same GitHub folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# 8. Recommend similar movies
def recommend_movies(movie_title, number=5):
    matches = movies.index[
        movies["title"] == movie_title
    ].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    scores = linear_kernel(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).ravel()

    # Highest similarity first
    ranked_indices = scores.argsort()[::-1]

    recommendations = []

    for index in ranked_indices:
        # Do not recommend the selected movie itself
        if index == movie_index:
            continue

        row = movies.iloc[index]

        recommendations.append({
            "title": row["title"],
            "genres": ", ".join(row["genres_list"]),
            "overview": row["overview"],
            "rating": row.get("vote_average", None),
            "similarity": float(scores[index])
        })

        if len(recommendations) >= number:
            break

    return recommendations


# 9. Build the website interface
st.success(f"Loaded {len(movies):,} movies successfully!")

movie_titles = sorted(movies["title"].unique().tolist())

default_index = (
    movie_titles.index("Avatar")
    if "Avatar" in movie_titles else 0
)

selected_movie = st.selectbox(
    "🎥 Select a movie you enjoy",
    movie_titles,
    index=default_index
)

number = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("✨ Recommend Movies", type="primary"):
    results = recommend_movies(selected_movie, number)

    if not results:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Movies similar to {selected_movie}")

        for movie in results:
            with st.container(border=True):
                st.subheader(f"🎬 {movie['title']}")

                if movie["genres"]:
                    st.caption(movie["genres"])

                st.write(
                    movie["overview"]
                    or "No plot summary available."
                )

                rating = movie["rating"]

                if pd.notna(rating):
                    st.write(f"⭐ TMDB rating: {rating}/10")

                st.caption(
                    "Content similarity: "
                    f"{movie['similarity'] * 100:.1f}%"
                )

st.divider()

st.caption(
    "Built using Python, Pandas, Streamlit and scikit-learn. "
    "Recommendations are based on movie metadata."
)

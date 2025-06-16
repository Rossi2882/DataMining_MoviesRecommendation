import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import random

#Wczytanie danych
BASE_DIR = Path(__file__).resolve().parent.parent
movies_path = BASE_DIR / 'data' / 'movies.csv'
ratings_path = BASE_DIR / 'data' / 'ratings.csv'

movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)
df = pd.merge(ratings_df, movies_df, on="movieId", how="left")

#Filtrowanie użytkowników
user_rating_counts = df["userId"].value_counts()
active_users = user_rating_counts[user_rating_counts >= 100].index
df = df[df["userId"].isin(active_users)]

#Statystyki opisowe
ratings_stats = df["rating"].describe()
ratings_per_movie = df.groupby("movieId")["rating"].count().describe()
ratings_per_user = df.groupby("userId")["rating"].count().describe()
general_info = {
    "Liczba unikalnych użytkowników": df["userId"].nunique(),
    "Liczba unikalnych filmów": df["movieId"].nunique(),
    "Liczba wszystkich ocen": len(df)
}

print("\n=== Statystyki ocen ===")
for k, v in ratings_stats.to_dict().items():
    print(f"{k}: {v}")
print("\n=== Informacje ogólne ===")
for k, v in general_info.items():
    print(f"{k}: {v}")
print("\n=== Oceny na film (statystyki) ===")
for k, v in ratings_per_movie.to_dict().items():
    print(f"{k}: {v}")
print("\n=== Oceny na użytkownika (statystyki) ===")
for k, v in ratings_per_user.to_dict().items():
    print(f"{k}: {v}")

#TF-IDF dla gatunków
movies_df["genres"] = movies_df["genres"].fillna("")
movies_df["genres_text"] = movies_df["genres"].str.replace("|", " ", regex=False)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df["genres_text"])
movie_id_to_index = pd.Series(movies_df.index, index=movies_df["movieId"])

#Funkcja do przewidywania zakrytej oceny
def predict_hidden_rating(user_id, df, tfidf_matrix, movie_id_to_index, movies_df, seed=None):
    user_ratings = df[df["userId"] == user_id]
    if len(user_ratings) < 2:
        return None, None, None

    hidden_row = user_ratings.sample(n=1, random_state=seed)
    hidden_movie_id = hidden_row.iloc[0]["movieId"]
    true_rating = hidden_row.iloc[0]["rating"]

    user_train = user_ratings[user_ratings["movieId"] != hidden_movie_id]
    hidden_idx = movie_id_to_index.get(hidden_movie_id)
    if pd.isna(hidden_idx):
        return hidden_movie_id, true_rating, None

    sim = cosine_similarity(tfidf_matrix[int(hidden_idx)], tfidf_matrix).flatten()
    similar_scores = []
    for _, row in user_train.iterrows():
        idx = movie_id_to_index.get(row["movieId"])
        if pd.isna(idx):
            continue
        similar_scores.append((row["rating"], sim[int(idx)]))

    if not similar_scores:
        return hidden_movie_id, true_rating, None

    weighted_sum = sum(r * s for r, s in similar_scores)
    sim_sum = sum(s for _, s in similar_scores)
    predicted_rating = weighted_sum / sim_sum if sim_sum else None

    return hidden_movie_id, true_rating, predicted_rating

#Funkcja do analizy ulubionych gatunków
def analyze_favorite_genres(user_id, df, movies_df, exclude_movie_id=None):
    user_ratings = df[df["userId"] == user_id][["movieId", "rating"]]
    if exclude_movie_id:
        user_ratings = user_ratings[user_ratings["movieId"] != exclude_movie_id]
    merged = pd.merge(user_ratings, movies_df[["movieId", "genres"]], on="movieId")
    genre_ratings = []

    for _, row in merged.iterrows():
        genres = row["genres"].split("|")
        for genre in genres:
            genre_ratings.append((genre, row["rating"]))

    genre_df = pd.DataFrame(genre_ratings, columns=["genre", "rating"])
    genre_stats = genre_df.groupby("genre")["rating"].agg(["count", "mean"])
    genre_stats = genre_stats[genre_stats["count"] >= 10]  #  dodane filtrowanie
    genre_stats = genre_stats.sort_values(by="mean", ascending=False)

    return genre_stats

#Ewaluacja 10 losowych użytkowników
print("\n=== Analiza 10 użytkowników: predykcja + ulubione gatunki ===")

selected_users = random.sample(list(active_users), 10)
true_vals = []
pred_vals = []

for uid in selected_users:
    hidden_movie_id, true_rating, predicted_rating = predict_hidden_rating(
        uid, df, tfidf_matrix, movie_id_to_index, movies_df, seed=None
    )

    print(f"\n Użytkownik ID: {uid}")

    if hidden_movie_id is not None and predicted_rating is not None:
        title = movies_df[movies_df["movieId"] == hidden_movie_id]["title"].values[0]
        error = abs(true_rating - predicted_rating)
        print(f"🎬 Zakryty film: {title} (movieId: {hidden_movie_id})")
        print(f"Rzeczywista ocena: {true_rating}, Przewidywana: {predicted_rating:.2f}, Błąd: {error:.2f}")
        true_vals.append(true_rating)
        pred_vals.append(predicted_rating)
    else:
        print(" Brak danych do przewidywania.")
        continue

    genre_stats = analyze_favorite_genres(uid, df, movies_df, exclude_movie_id=hidden_movie_id)
    print("\n  Top 5 ulubionych gatunków:")
    print(genre_stats.head(5).round(2))

#Ogólna ewaluacja: MAE, CI, permutacyjny test
if true_vals and pred_vals:
    mae = np.mean([abs(t - p) for t, p in zip(true_vals, pred_vals)])
    print(f"\n  Średni błąd bezwzględny (MAE): {mae:.4f}")

    def bootstrap_ci(data, n_iterations=1000, ci=0.95):
        means = []
        for _ in range(n_iterations):
            sample = resample(data)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return np.mean(means), (lower, upper)

    errors = [abs(t - p) for t, p in zip(true_vals, pred_vals)]
    mean_mae, (ci_low, ci_high) = bootstrap_ci(errors)
    print(f"  Przedział ufności dla MAE (95%): ({ci_low:.4f}, {ci_high:.4f})")

    def permutation_test_model_vs_random(true_ratings, predicted_ratings, n_iter=1000):
        actual_mae = np.mean([abs(t - p) for t, p in zip(true_ratings, predicted_ratings)])
        n_better = 0
        for _ in range(n_iter):
            random_preds = np.random.uniform(0.5, 5.0, size=len(true_ratings))
            random_mae = np.mean([abs(t - r) for t, r in zip(true_ratings, random_preds)])
            if random_mae <= actual_mae:
                n_better += 1
        p_value = n_better / n_iter
        return actual_mae, p_value

    mae, p_val = permutation_test_model_vs_random(true_vals, pred_vals)
    print(f"\n  Permutacyjny test statystyczny:")
    print(f"MAE modelu: {mae:.4f}")
    print(f"p-wartość: {p_val:.4f}")
    if p_val < 0.05:
        print(" Model jest statystycznie lepszy niż losowe przewidywanie (p < 0.05)")
    else:
        print(" Model NIE jest istotnie lepszy od losowego (p >= 0.05)")
else:
    print("\nBrak danych do podsumowania.")

print("\n=== Rekomendacje z ulubionych gatunków użytkownika ===")

def recommend_from_favorite_genres(user_id, df, movies_df, top_n_genres=3, min_rating=3.5, min_votes=20, top_k=10):
    # Filmy już ocenione przez użytkownika
    seen_movie_ids = set(df[df["userId"] == user_id]["movieId"])

    # Ulubione gatunki użytkownika
    genre_stats = analyze_favorite_genres(user_id, df, movies_df)
    top_genres = genre_stats.head(top_n_genres).index.tolist()

    # Oblicz średnie oceny i liczby ocen dla wszystkich filmów
    movie_stats = df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    movie_stats.columns = ["movieId", "avg_rating", "num_ratings"]

    # Filtrujemy tylko filmy spełniające kryteria
    filtered_movies = pd.merge(movies_df, movie_stats, on="movieId")
    filtered_movies = filtered_movies[
        (filtered_movies["avg_rating"] >= min_rating) &
        (filtered_movies["num_ratings"] >= min_votes) &
        (~filtered_movies["movieId"].isin(seen_movie_ids))
    ]

    #Filmy należące do ulubionych gatunków
    def has_genre(row):
        return any(g in row["genres"].split("|") for g in top_genres)

    recommended = filtered_movies[filtered_movies.apply(has_genre, axis=1)]
    recommended = recommended.sort_values(by="avg_rating", ascending=False).head(top_k)

    return recommended[["title", "genres", "avg_rating", "num_ratings"]]

# Dla każdego z wcześniej wybranych 10 użytkowników
for uid in selected_users:
    print(f"\n Użytkownik ID: {uid}")
    recs = recommend_from_favorite_genres(uid, df, movies_df)

    if recs.empty:
        print("Brak filmów do polecenia w ulubionych gatunkach.")
    else:
        print("Proponowane filmy z ulubionych gatunków:")
        for i, row in recs.iterrows():
            print(f"- {row['title']} ({row['genres']}) — Ocena: {row['avg_rating']:.2f}, Głosów: {int(row['num_ratings'])}")

for uid in selected_users:
    print(f"\n Użytkownik ID: {uid}")
    recs = recommend_from_favorite_genres(uid, df, movies_df)

    if recs.empty:
        print("Brak filmów do polecenia w ulubionych gatunkach.")
    else:
        print("Proponowane filmy z ulubionych gatunków:")
        for i, row in recs.iterrows():
            print(f"- {row['title']} ({row['genres']}) — Ocena: {row['avg_rating']:.2f}, Głosów: {int(row['num_ratings'])}")

#Histogramy średnich ocen gatunków
import matplotlib.pyplot as plt

def plot_genre_rating_histograms(user_ids, df, movies_df):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle("🎬 Top 5 ocenianych gatunków dla 10 losowych użytkowników", fontsize=18, y=1.05)

    for i, uid in enumerate(user_ids):
        row = i // 5
        col = i % 5
        ax = axes[row, col]

        genre_stats = analyze_favorite_genres(uid, df, movies_df)
        if genre_stats.empty:
            ax.axis("off")
            continue

        top_genres = genre_stats.head(5)
        top_genres["mean"].plot(kind="bar", ax=ax)
        ax.set_title(f"Użytkownik {uid}", fontsize=10)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Średnia ocena")
        ax.set_xlabel("Gatunek")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

plot_genre_rating_histograms(selected_users, df, movies_df)

def simulate_recommendation_hit_rate(user_id, df, movies_df, top_k=500, test_fraction=0.2, min_rated=15):
    user_ratings = df[df["userId"] == user_id]
    if len(user_ratings) < min_rated:
        return None

    test_size = max(1, int(len(user_ratings) * test_fraction))
    test_set = user_ratings.sample(n=test_size, random_state=42)
    train_set = user_ratings[~user_ratings.index.isin(test_set.index)]

    #Rekomendacje na podstawie train_set
    df_train = df[df["userId"] != user_id].copy()
    df_train = pd.concat([df_train, train_set], axis=0)

    recs = recommend_from_favorite_genres(user_id, df_train, movies_df, top_k=top_k)
    if recs.empty:
        return 0.0

    recommended_ids = set(movies_df[movies_df["title"].isin(recs["title"])]["movieId"])
    test_movie_ids = set(test_set["movieId"])

    hits = len(test_movie_ids & recommended_ids)
    hit_rate = hits / len(test_movie_ids)

    return hit_rate


print("\n===  Ewaluacja rekomendacji przez symulację ukrycia ocen (Hit Rate@500) ===")

hit_rates = []
for uid in selected_users:
    hr = simulate_recommendation_hit_rate(uid, df, movies_df, top_k=500)
    if hr is not None:
        hit_rates.append(hr)
        print(f"User {uid} — Hit Rate@500: {hr:.2f}")
    else:
        print(f"User {uid} — Za mało danych do oceny")

if hit_rates:
    avg_hr = np.mean(hit_rates)
    print(f"\n Średni Hit Rate@10: {avg_hr:.4f}")
else:
    print("Brak wystarczających danych do podsumowania.")
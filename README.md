# Movie Recommendation System Based on Data Analysis

Project created for the **Data Mining** course. The goal was to develop a simple yet multi-faceted movie recommendation system based on user ratings and movie genre descriptions.

## Repository Contents

├── data/ # Input data files (.csv)
│ ├── movies.csv
│ └── ratings.csv
├── src/
│ └── MoviesRecommendation.py
├── .gitignore
├── README.md

## Data

The data comes from the **MovieLens** dataset and includes:
- `movies.csv` — list of movies with titles and genres
- `ratings.csv` — ratings given by users (scale 0.5–5.0)

## What does the project do?

The project implements several steps of data analysis and recommendation:

### Preliminary Analysis:
- Filtering active users (minimum 100 ratings)
- Statistics on the number of ratings per movie/user
- Transformation of movie genres using TF-IDF

### Hidden Rating Prediction:
- Hiding one random user rating
- Rating prediction based on movie genre similarity
- Calculation of the MAE (Mean Absolute Error) and its confidence interval
- Permutation statistical test comparing the model with random guessing

### Favorite Genres Analysis:
- Detecting a user's highest-rated genres
- Histograms of average genre ratings for 10 random users

### Recommendations:
- Recommending movies from the user's favorite genres
- Filtering by average rating and number of ratings
- Evaluation using **Hit Rate@500**

## Example Metrics

- MAE (Mean Absolute Error): ~0.60–0.75
- Hit Rate@500: ~0.20–0.35
- Permutation test indicates that the model usually performs better than random guessing

## ▶️ How to run?

1. Place the `movies.csv` and `ratings.csv` files in the `data/` directory
2. Navigate to the `src/` folder
3. Run:

```bash
python MoviesRecommendation.py

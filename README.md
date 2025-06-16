# System rekomendacji filmów na podstawie analizy danych

Projekt stworzony w ramach przedmiotu **Eksploracja Danych**. Celem było opracowanie prostego, ale wieloaspektowego systemu rekomendacji filmów, bazującego na danych z ocen użytkowników i opisów gatunków filmowych.

## Zawartość repozytorium

├── data/ # Pliki z danymi wejściowymi (.csv)
│ ├── movies.csv
│ └── ratings.csv
├── src/
│ └── MoviesRecommendation.py
├── .gitignore
├── README.md

## Dane

Dane pochodzą z zestawu **MovieLens** i zawierają:
- `movies.csv` — lista filmów z tytułami i gatunkami
- `ratings.csv` — oceny wystawione przez użytkowników (skala 0.5–5.0)

##  Co robi projekt?

Projekt implementuje kilka kroków analizy danych i rekomendacji:

###  Wstępna analiza:
- Filtrowanie aktywnych użytkowników (minimum 100 ocen)
- Statystyki liczby ocen na film/użytkownika
- Przekształcenie gatunków filmowych z użyciem TF-IDF

###  Predykcja ukrytej oceny:
- Ukrycie jednej losowej oceny użytkownika
- Predykcja oceny na podstawie podobieństwa gatunków filmów
- Obliczenie błędu MAE i jego przedziału ufności
- Permutacyjny test statystyczny porównujący model z losowym zgadywaniem

###  Analiza ulubionych gatunków:
- Wykrywanie najlepiej ocenianych gatunków przez użytkownika
- Histogramy średnich ocen gatunków dla 10 losowych użytkowników

###  Rekomendacje:
- Polecanie filmów z ulubionych gatunków użytkownika
- Filtrowanie według średniej oceny i liczby ocen
- Ewaluacja za pomocą **Hit Rate@500**

##  Przykładowe metryki

- MAE (średni błąd bezwzględny): ~0.60–0.75
- Hit Rate@500: ~0.20–0.35
- Permutacyjny test wskazuje, że model zwykle wypada lepiej niż losowe przewidywanie

## ▶️ Jak uruchomić?

1. Umieść pliki `movies.csv` i `ratings.csv` w katalogu `data/`
2. Przejdź do folderu `src/`
3. Uruchom:

```bash
python MoviesRecommendation.py

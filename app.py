from collections import Counter
import random
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Sample historical data
data = [
    [12, 25, 42, 46, 63, 22],
    [35, 25, 46, 51, 63, 5], 
    [12, 42, 46, 51, 65, 19],
    data = [
    [47, 54, 57, 60, 65, 19],
    [13, 31, 51, 55, 66, 23],
    [9, 35, 54, 63, 64, 1],
    [12, 30, 39, 64, 67, 22],
    [12, 26, 27, 43, 47, 5],
    [11, 26, 35, 40, 43, 24],
    [19, 30, 37, 44, 46, 22],
    [2, 22, 46, 56, 67, 25],
    [1, 7, 46, 47, 63, 7],
    [10, 25, 51, 52, 63, 1],
    [10, 12, 22, 36, 50, 4],
    [1, 12, 20, 33, 66, 21],
    [14, 16, 26, 41, 57, 24],
    [16, 27, 59, 62, 63, 23],
    [6, 20, 29, 46, 68, 8],
    [2, 21, 26, 40, 42, 9],
    [3, 7, 41, 48, 67, 14],
    [8, 11, 19, 24, 46, 5],
    [10, 27, 30, 43, 47, 6],
    [22, 30, 37, 44, 45, 18],
    [3, 12, 28, 58, 62, 24],
    [9, 25, 27, 53, 66, 5],
    [18, 22, 25, 30, 35, 22],
    [11, 19, 29, 63, 68, 25],
    [33, 38, 42, 64, 65, 20],
    [9, 14, 20, 23, 63, 1],
    [1, 12, 30, 50, 55, 10],
    [1, 26, 32, 46, 51, 13],
    [1, 29, 45, 47, 51, 7],
    [25, 38, 42, 66, 67, 19],
    [3, 7, 14, 26, 52, 21],
    [4, 13, 35, 61, 69, 4],
    [2, 10, 45, 46, 59, 2],
    [4, 6, 25, 55, 68, 26],
    [23, 24, 48, 61, 64, 4],
    [20, 22, 26, 28, 63, 5],
    [1, 19, 54, 63, 64, 15],
    [25, 30, 32, 33, 55, 20],
    [18, 25, 38, 56, 61, 19],
    [3, 4, 12, 22, 28, 16],
    [1, 13, 19, 49, 66, 7],
    [1, 25, 27, 38, 62, 13],
    [19, 26, 39, 65, 68, 12],
    [9, 11, 17, 19, 55, 1],
    [15, 22, 44, 49, 63, 26],
    [32, 34, 37, 39, 47, 3],
    [2, 23, 42, 61, 68, 26],
    [19, 21, 37, 50, 65, 26],
    [10, 26, 27, 48, 52, 12],
    [10, 15, 21, 67, 69, 3],
    [1, 42, 51, 61, 63, 17],
    [6, 13, 20, 35, 54, 22],
    [26, 29, 37, 49, 63, 22],
    [18, 42, 44, 62, 65, 23],
    [14, 25, 37, 57, 67, 16],
    [23, 24, 33, 51, 64, 5],
    [5, 23, 30, 62, 63, 9]
]


def weighted_frequency_analysis(data):
    recent_data = data[-10:]
    weighted_numbers = [num for sublist in recent_data for num in sublist[:-1]]
    return Counter(weighted_numbers)

def bigram_analysis(data):
    bigrams = []
    for sublist in data:
        for i in range(4):
            bigrams.append((sublist[i], sublist[i+1]))
    return Counter(bigrams)

def train_ml_model(data):
    X = [sublist[:-1] for sublist in data[:-1]]
    y = [sublist[:-1] for sublist in data[1:]]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def predict_next_combination(data, model):
    weighted_counts = weighted_frequency_analysis(data)
    bigram_counts = bigram_analysis(data)
    combined_counts = weighted_counts + bigram_counts
    common_numbers = [item[0] for item in combined_counts.most_common(5)]
    
    # Introduce some randomness to get varied predictions
    random.shuffle(common_numbers)
    
    ml_prediction = model.predict([common_numbers])
    return list(map(int, ml_prediction[0]))

def main():
    st.title("Lottery Number Predictor")
    
    if st.button("Predict Top 5 Combinations"):
        predictions = []
        model = train_ml_model(data)
        for _ in range(5):
            prediction = predict_next_combination(data, model)
            predictions.append(prediction)
            if prediction in data:
                data.remove(prediction)
        
        for i, pred in enumerate(predictions, 1):
            st.write(f"Prediction {i}: {pred}")

if __name__ == "__main__":
    main()

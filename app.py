from collections import Counter
import random
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Sample historical data
data = [
    [12, 25, 42, 46, 63, 22],
    [35, 25, 46, 51, 63, 5], 
    [12, 42, 46, 51, 65, 19],
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
    [10, 12, 22, 36, 50, 4]
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

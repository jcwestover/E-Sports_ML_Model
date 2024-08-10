import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import precision_score, f1_score, accuracy_score, mean_squared_error, log_loss, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import itertools
import altair as alt

@st.cache_resource()
def load_model():
    return pickle.load(open('xgb_model.pkl', 'rb'))

model = load_model()

st.title("Halo Infinite ML Model")

# 'Cnasty703', 'HafenNation', 'Steelblade01', 'YungJaguar', 'mcddp15', 'zE tthrilla', 'zE eskky', time_Friday', 'time_Monday', 'time_Saturday', 'time_Sunday', 'time_Thursday', 'time_Tuesday', 'time_Wednesday', 'gamemode_Capture the Flag', 'gamemode_Extraction', 'gamemode_King of the Hill', 'gamemode_Oddball', 'gamemode_Strongholds', 'gamemode_Team Slayer', 'game_map_Aquarious', 'game_map_Argyle', 'game_map_Cliffhangar', 'game_map_Empyrean', 'game_map_Forbidden', 'game_map_Forest', 'game_map_Interference', 'game_map_Live Fire', 'game_map_Recharge', 'game_map_Solitude', 'game_map_Streets'

users = ['Cnasty703', 'HafenNation', 'mcddp15', 'Steelblade01', 'YungJaguar', 'zE eskky', 'zE tthrilla']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
modes = ['Capture the Flag', 'Extraction', 'King of the Hill', 'Oddball', 'Strongholds', 'Team Slayer']
maps = ['Aquarious', 'Argyle', 'Cliffhangar', 'Empyrean', 'Forbidden', 'Forest', 'Interference', 'Live Fire', 'Recharge', 'Solitude', 'Streets']

selected_users = st.multiselect('Select Users:', users,  max_selections=4, placeholder='Select up to 4 users')
selected_day = st.selectbox('Select Day:', days, index=None)
selected_mode = st.selectbox('Select Game Mode:', modes, index=None)
selected_map = st.selectbox('Select Map:', maps, index=None)

initial_data = pd.DataFrame(0, index=[0], columns=[
        'Cnasty703', 'HafenNation', 'Steelblade01', 'YungJaguar', 'mcddp15', 'zE tthrilla', 'zE eskky', 'time_Friday', 'time_Monday', 'time_Saturday', 'time_Sunday', 'time_Thursday', 'time_Tuesday', 'time_Wednesday', 'gamemode_Capture the Flag', 'gamemode_Extraction', 'gamemode_King of the Hill', 'gamemode_Oddball', 'gamemode_Strongholds', 'gamemode_Team Slayer', 'game_map_Aquarious', 'game_map_Argyle', 'game_map_Cliffhangar', 'game_map_Empyrean', 'game_map_Forbidden', 'game_map_Forest', 'game_map_Interference', 'game_map_Live Fire', 'game_map_Recharge', 'game_map_Solitude', 'game_map_Streets'
    ])

input_data = pd.DataFrame(0, index=[0], columns=[
        'Cnasty703', 'HafenNation', 'Steelblade01', 'YungJaguar', 'mcddp15', 'zE tthrilla', 'zE eskky', 'time_Friday', 'time_Monday', 'time_Saturday', 'time_Sunday', 'time_Thursday', 'time_Tuesday', 'time_Wednesday', 'gamemode_Capture the Flag', 'gamemode_Extraction', 'gamemode_King of the Hill', 'gamemode_Oddball', 'gamemode_Strongholds', 'gamemode_Team Slayer', 'game_map_Aquarious', 'game_map_Argyle', 'game_map_Cliffhangar', 'game_map_Empyrean', 'game_map_Forbidden', 'game_map_Forest', 'game_map_Interference', 'game_map_Live Fire', 'game_map_Recharge', 'game_map_Solitude', 'game_map_Streets'
    ])
    
if st.button('Predict Win Probability'):
    for user in selected_users:
        input_data[user] = 1

    input_data[f'time_{selected_day}'] = 1
    input_data[f'gamemode_{selected_mode}'] = 1
    input_data[f'game_map_{selected_map}'] = 1


    win_probability = model.predict_proba(input_data)[:, 1][0]

    st.write(f'Win probability: {win_probability * 100:.2f}%')

# others to add
# win probability vs selected features
if st.button('Find Best Team'):
    player_combinations = []
    probabilities = []

    for combo in itertools.combinations(users, 4):
        input_data = initial_data.copy()
    
    for user in combo:
        input_data[user] = 1

    input_data[f'time_{selected_day}'] = 1
    input_data[f'gamemode_{selected_mode}'] = 1
    input_data[f'game_map_{selected_map}'] = 1
        
    win_probability = model.predict_proba(input_data)[:, 1][0]

    player_combinations.append(', '.join(combo))
    probabilities.append(win_probability)

    df = pd.DataFrame({
        'Team': player_combinations,
        'Win Probability': probabilities
    })

    df = df.sort_values(by='Win Probability', ascending=False)

    st.write(f'Best Team: {df.iloc[0]['Team']}')
    st.write(f'Win Probability: {df.iloc[0]['Win Probability'] * 100:.2f}')

# more about this model button
# feature importance plot

# if st.button('Show Feature Importance'):
#     feature_importances = model.feature_importances_
#     features = input_data.columns

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.barh(features, feature_importances)
#     plt.title('Feature Importance')
#     plt.xlabel('Importance')
#     st.pyplot(plt)

# confusion matrix

# if st.button('Show Confusion Matrix'):
#     y_pred = model.predict(X_test)  # Assuming you have X_test and y_test
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     st.pyplot(plt)

# Using "with" notation
import sys
import pandas as pd
import numpy as np

sys.path.append(r"C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data\main")

import main

print(main.data.shape)
print(main.data.head())

print(pd.value_counts(main.data['Gender']))

map_gender = {gender: idx for idx, gender in enumerate(main.data['Gender'].unique())}

main.data['Gender'] = main.data['Gender'].map(map_gender)

print(main.data['Gender'].head())
print(map_gender)

print(pd.value_counts(main.data['City']))
print(pd.value_counts(main.data['Profession']))

map_profession = {profession: idx for idx, profession in enumerate(main.data['Profession'].unique())}
main.data['Profession'] = main.data['Profession'].map(map_profession)
print(main.data['Profession'].head())
print(map_profession)

map_sleep_duration = {sleep_duration: idx for idx, sleep_duration in enumerate(main.data['Sleep Duration'].unique())}
main.data['Sleep Duration'] = main.data['Sleep Duration'].map(map_sleep_duration)
print(main.data['Sleep Duration'].head())
print(map_sleep_duration)

print(pd.value_counts(main.data['Dietary Habits']))
map_dietary_habits = {dietary_habits: idx for idx, dietary_habits in enumerate(main.data['Dietary Habits'].unique())}
main.data['Dietary Habits'] = main.data['Dietary Habits'].map(map_dietary_habits)
print(main.data['Dietary Habits'].head())
print(map_dietary_habits)

print(pd.value_counts(main.data['Have you ever had suicidal thoughts ?']))
map_suicide = {suicide: idx for idx, suicide in enumerate(main.data['Have you ever had suicidal thoughts ?'].unique())}
main.data['Have you ever had suicidal thoughts ?'] = main.data['Have you ever had suicidal thoughts ?'].map(map_suicide)
print(main.data['Have you ever had suicidal thoughts ?'].head())
print(map_suicide)

print(pd.value_counts(main.data['Family History of Mental Illness']))
map_family_history = {family: idx for idx, family in enumerate(main.data['Family History of Mental Illness'].unique())}
main.data['Family History of Mental Illness'] = main.data['Family History of Mental Illness'].map(map_family_history)
print(main.data['Family History of Mental Illness'].head())
print(map_family_history)

print(pd.value_counts(main.data['Work Pressure']))

print(main.data.head())
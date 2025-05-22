# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Project: Building a Rainfall Prediction Classifier

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# %% [markdown]
# ## Load the data
#

# %% [markdown]
# The dataset contains observations of weather metrics for each day from 2008 to 2017, and includes the following fields:
#
# | Field         | Description                                           | Unit            | Type   |
# | :------------ | :---------------------------------------------------- | :-------------- | :----- |
# | Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
# | Location      | Location of the Observation                           | Location        | object |
# | MinTemp       | Minimum temperature                                   | Celsius         | float  |
# | MaxTemp       | Maximum temperature                                   | Celsius         | float  |
# | Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
# | Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
# | Sunshine      | Amount of bright sunshine                             | hours           | float  |
# | WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
# | WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
# | WindDir9am    | Wind direction averaged over 10 minutes prior to 9am  | Compass Points  | object |
# | WindDir3pm    | Wind direction averaged over 10 minutes prior to 3pm  | Compass Points  | object |
# | WindSpeed9am  | Wind speed averaged over 10 minutes prior to 9am      | Kilometers/Hour | float  |
# | WindSpeed3pm  | Wind speed averaged over 10 minutes prior to 3pm      | Kilometers/Hour | float  |
# | Humidity9am   | Humidity at 9am                                       | Percent         | float  |
# | Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
# | Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
# | Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
# | Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
# | Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
# | Temp9am       | Temperature at 9am                                    | Celsius         | float  |
# | Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
# | RainToday     | If there was at least 1mm of rain today               | Yes/No          | object |
# | RainTomorrow  | If there is at least 1mm of rain tomorrow             | Yes/No          | object |
#

# %%
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

# %%
df.count()

# %% [markdown]
# ### Drop all rows with missing values
#

# %%
df = df.dropna()
df.info()

# %%
df.columns

# %% [markdown]
# ## Data leakage considerations
#

# %% [markdown]
# ## Points to note - 1
# List some of the features that would be inefficient in predicting tomorrow's rainfall. There will be a question in the quiz that follows based on this observation.

# %% [markdown]
# RainToday wouldn't be ideal as it needs a full day of data.

# %% [markdown]
# If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday, then we can legitimately utilize all of the available features. 

# %%
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# %% [markdown]
# ## Data Granularity
#

# %% [markdown]
# Would the weather patterns have the same predictability in vastly different locations in Australia? Probably not.  
# The chance of rain in one location can be much higher than in another. 
# Using all of the locations requires a more complex model as it needs to adapt to local weather patterns.  
# Let's see how many observations we have for each location, and see if we can reduce our attention to a smaller region.

# %% [markdown]
# ## Location selection
#
# Watsonia is only 15 km from Melbourne, and the Melbourne Airport is only 18 km from Melbourne.  
# Let's group these three locations together and use only their weather data to build our localized prediction model.  
# Because there might still be some slight variations in the weather patterns we'll keep `Location` as a categorical variable.

# %%
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# %% [markdown]
# We still have 7557 records, which should be enough to build a reasonably good model.  

# %% [markdown]
# ## Extracting a seasonality feature
#
# Now consider the `Date` column. We expect the weather patterns to be seasonal, having different predictablitiy levels in winter and summer for example.  
# There may be some variation with `Year` as well, but we'll leave that out for now.
# Let's engineer a `Season` feature from `Date` and drop `Date` afterward, since it is most likely less informative than season. 
# An easy way to do this is to define a function that assigns seasons to given months, then use that function to transform the `Date` column.
#

# %%
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


# %% [markdown]
# ## Exercise 1: Map the dates to seasons and drop the Date column
#

# %%
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

df=df.drop(columns=['Date'])
df['Season'].value_counts()

# %% [markdown]
# ## Exercise 2. Define the feature and target dataframes
#

# %%
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

# %% [markdown]
# ## Exercise 3. How balanced are the classes?
#

# %%
y.value_counts()

# %% [markdown]
# The classes are imbalanced, there are roughly 3x more days without rain than with rain.

# %% [markdown]
# ## Exercise 4. What can you conclude from these counts?
# - How often does it rain annually in the Melbourne area?
# - How accurate would you be if you just assumed it won't rain every day?
# - Is this a balanced dataset?
# - Next steps?
#

# %% [markdown]
# - TODO
# - 75% accurate
# - no
# - TODO

# %% [markdown]
# ## Exercise 5. Split data into training and test sets, ensuring target stratification
#

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# ## Define preprocessing transformers for numerical and categorical features
# ## Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features

# %%
numeric_features = X_train.select_dtypes(include=['float64']).columns.tolist()  
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# %% [markdown]
# ### Define separate transformers for both feature types and combine them into a single preprocessing transformer
#

# %%
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# %% [markdown]
# ## Exercise 7. Combine the transformers into a single preprocessing column transformer

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %% [markdown]
# ## Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier

# %%
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %% [markdown]
# ### Define a parameter grid to use in a cross validation grid search model optimizer
#

# %%
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# %% [markdown]
# ## Perform grid search cross-validation and fit the best model to the training data
# ### Select a cross-validation method, ensuring target stratification during validation
#

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True)

# %% [markdown]
# ## Exercise 9. Instantiate and fit GridSearchCV to the pipeline

# %%
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)  
grid_search.fit(X_train, y_train)



# %% [markdown]
# ### Print the best parameters and best crossvalidation score
#

# %%
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# %% [markdown]
# ## Exercise 10. Display your model's estimated score
#

# %%
test_score = grid_search.score(X_test, y_test)  
print("Test set score: {:.2f}".format(test_score))


# %% [markdown]
# ## Exercise 11. Get the model predictions from the grid search estimator on the unseen data
#

# %%
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ## Exercise 12. Print the classification report
#

# %%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## Exercise 13. Plot the confusion matrix 
#

# %%
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# %%
1097 / (1097 + 57)

# %%
tn, fp, fn, tp = conf_matrix.ravel()
tp / (tp + fn)

# %% [markdown]
# ## Points to note - 2
# What is the true positive rate? There will be a question on this in the assignment that follows.
#

# %% [markdown]
# The true positive rate is 1097 / (1097 + 57) = 0.95

# %% [markdown]
# ## Feature importances
#

# %% [markdown]
# ## Exercise 14. Extract the feature importances
#

# %%
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


# %% [markdown]
# Now let's extract the feature importances and plot them as a bar graph.
#

# %%
# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# %% [markdown]
# ## Point to note - 3
# Identify the most important feature for predicting whether it will rain based on the feature importance bar graph. There will be a question on this in the assignment that follows.
#

# %% [markdown]
# Humidity at 3pm

# %% [markdown]
# ## Try another model
# #### Some thoughts.
# In practice you would want to try out different models and even revisit the data analysis to improve
# your model's performance. Maybe you can engineer better features, drop irrelevant or redundant ones, project your data onto a dimensional feature space, or impute missing values to be able to use more data. You can also try a larger set of parameters to define you search grid, or even engineer new features using cluster analysis. You can even include the clustering algorithm's hyperparameters in your search grid!
#
# With Scikit-learn's powerful pipeline and GridSearchCV classes, this is easy to do in a few steps.
#

# %% [markdown]
# ## Exercise 15. Update the pipeline and the parameter grid
#

# %%
# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ###  Compare the results to your previous model.
#

# %%
print(classification_report(y_test, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# What can you conclude about the model performances? 
#
# They are similar.

# %% [markdown]
# ## Points to note - 4
# Compare the accuracy and true positive rate of rainfall predictions between the LogisticRegression model and the RandomForestClassifier model.

# %%
tn, fp, fn, tp = conf_matrix.ravel()
tp / (tp + fn)

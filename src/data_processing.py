import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Create a bar plot for a categorical feature and a line plot for the target variable.
def plot_categorical_with_target(df, feature, target='price'):
    # Group by the feature to calculate count and mean target
    grouped = df.groupby(feature).agg(count=(feature, 'size'), price_mean=(target, 'mean')).reset_index()
    # Create a subplot with 2 rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"Distribution of {feature}", f"Average {target} by {feature}"])
    # Bar plot for distribution (count)
    fig.add_trace(
        go.Bar(
            x=grouped[feature],
            y=grouped['count'],
            name='Count',
            marker=dict(color='blue')
        ),
        row=1, col=1
    )
    # Line plot for average target
    fig.add_trace(
        go.Scatter(
            x=grouped[feature],
            y=grouped['price_mean'],
            name=f'Average {target}',
            mode='lines+markers',
            marker=dict(color='orange')
        ),
        row=2, col=1
    )
    # Update layout
    fig.update_layout(
        height=600, width=800,
        showlegend=False,
        title_text=f"Analysis of {feature} and its relation to {target}",
        xaxis_title=feature,
        yaxis_title="Count",
        yaxis2_title=f"Average {target}"
    )
    
    fig.show()


# Categorical features with type int64 conversion to category
def convert_to_category(df, features):
    df_copy = df.copy()
    for feature in features:
        df_copy[feature] = df_copy[feature].astype('category')
    return df_copy


# Check if there are duplicates in the dataset
def handle_duplicates(df):
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Number of duplicates: {duplicates}. Shape DF: {df.shape}")
        df_cleaned = df.drop_duplicates()
        print(f"Number of removed duplicates: {duplicates}. New shape DF: {df_cleaned.shape}")
    else:
        print("No duplicates found!")
        df_cleaned = df

    return df_cleaned


# Implements One-Hot Encoding on the categorical features
def apply_encoding(df):

    cat_features = df.select_dtypes(include='category').columns
    encoding_features = [feature for feature in cat_features if df[feature].nunique() > 2]

    df_encoded = pd.get_dummies(df, columns=encoding_features, drop_first=True, dtype=int)

    return df_encoded


# StandardScaler
def feature_scaling_standardization(X_train, X_test):
    continuous_features = ['area'] 

    continuous_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    preprocessing = ColumnTransformer(
        transformers = [
            ('continuos', continuous_transformer, continuous_features)
        ],
        remainder = 'passthrough'
    )

    X_train_scaled = preprocessing.fit_transform(X_train)
    X_test_scaled = preprocessing.transform(X_test)

    feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_scaled, X_test_scaled
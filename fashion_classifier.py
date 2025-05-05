import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(file_path):
    """
    Load the fashion dataset and display basic information
    """
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)
    
    print("\n=== Dataset Overview ===")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Extract fashion categories from the Tags column
    # We'll use the first tag in each row as the fashion style/category
    df['fashion_category'] = df['Tags'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0])
    
    print("\nFashion Categories Distribution:")
    category_counts = df['fashion_category'].value_counts()
    print(category_counts)
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for ML modeling
    """
    print("\n=== Preprocessing Data ===")
    
    # Extract features from QuestionText
    # We'll use regex to identify important elements from questions
    df['occasion'] = df['QuestionText'].str.extract(r'for\s+(\w+(?:\s+\w+)?)\s+in')
    df['season'] = df['QuestionText'].str.extract(r'in\s+(\w+)\??$')
    
    # Extract item names - fixed to properly handle regex results
    df['item'] = df['QuestionText'].str.extract(r'What\s+(\w+)')
    # Handle "Which" pattern
    which_items = df['QuestionText'].str.extract(r'Which\s+(\w+)')
    df.loc[df['item'].isna(), 'item'] = which_items.loc[df['item'].isna(), 0]
    # Handle "Describe a" pattern
    describe_items = df['QuestionText'].str.extract(r'Describe\s+a\s+(\w+)')
    df.loc[df['item'].isna(), 'item'] = describe_items.loc[df['item'].isna(), 0]
    
    # Parse AnswerText to extract style, color, and material
    df['style'] = df['AnswerText'].str.split().str[0]
    df['color'] = df['AnswerText'].str.split().str[1]
    df['material'] = df['AnswerText'].str.split().str[2]
    
    print("\nFeatures extracted:")
    print(df[['occasion', 'season', 'item', 'style', 'color', 'material']].head())
    
    # Handle missing values
    for col in ['occasion', 'season', 'item', 'style', 'color', 'material']:
        df[col] = df[col].fillna('unknown')
        print(f"Column '{col}' has {df[col].isna().sum()} missing values after imputation")
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    
    # For simplicity, we'll use one-hot encoding for categorical features
    features_to_encode = ['occasion', 'season', 'item', 'color', 'material']
    
    # Use CountVectorizer for simplicity and sparsity handling
    encoders = {}
    encoded_features = pd.DataFrame()
    
    for feature in features_to_encode:
        encoders[feature] = CountVectorizer()
        feature_encoded = encoders[feature].fit_transform(df[feature].astype(str))
        feature_names = [f"{feature}_{x}" for x in encoders[feature].get_feature_names_out()]
        encoded_df = pd.DataFrame.sparse.from_spmatrix(feature_encoded, columns=feature_names, index=df.index)
        encoded_features = pd.concat([encoded_features, encoded_df], axis=1)
    
    # Also encode the style column separately as it will be our target
    style_encoder = LabelEncoder()
    df['fashion_category_encoded'] = style_encoder.fit_transform(df['fashion_category'])
    
    print(f"Total number of encoded features: {encoded_features.shape[1]}")
    
    return encoded_features, df['fashion_category_encoded'], df['fashion_category'], style_encoder, encoders, df

def train_and_evaluate_models(X, y, category_names):
    """
    Train multiple classifiers and evaluate their performance
    """
    print("\n=== Training Models ===")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Define and train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', multi_class='ovr'),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=category_names, zero_division=0)
        print(report)
    
    # Determine best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    print(f"\n=== Best Model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f}) ===")
    
    return results[best_model_name]['model'], results

def recommend_fashion(query, model, style_encoder, feature_encoders, original_df):
    """
    Recommend a fashion category based on user input
    Returns the category and a confidence score
    """
    # Parse the user query to extract features
    features = {}
    
    # Try to extract occasion
    occasion_match = re.search(r'for\s+(\w+(?:\s+\w+)?)', query)
    if occasion_match:
        features['occasion'] = occasion_match.group(1).lower()
    else:
        features['occasion'] = "casual"  # Default
    
    # Try to extract season
    season_match = re.search(r'in\s+(\w+)', query)
    if season_match:
        features['season'] = season_match.group(1).lower()
    else:
        features['season'] = "summer"  # Default
    
    # Try to extract item
    item_match = re.search(r'(dress|shirt|pants|jeans|t-shirt|skirt|jacket|coat|hoodie|sweater|blouse|shoes|boots|sneakers|sandals|accessories)', query.lower())
    if item_match:
        features['item'] = item_match.group(1).lower()
    else:
        features['item'] = "unknown"
    
    # Try to extract color preference
    color_match = re.search(r'(black|white|red|blue|green|yellow|pink|purple|orange|gray|beige|brown)', query.lower())
    if color_match:
        features['color'] = color_match.group(1).lower()
    else:
        features['color'] = "unknown"
    
    # Try to extract material preference
    material_match = re.search(r'(cotton|silk|wool|leather|polyester|denim|linen|velvet|suede|nylon)', query.lower())
    if material_match:
        features['material'] = material_match.group(1).lower()
    else:
        features['material'] = "unknown"
    
    # Prepare input for prediction
    input_features = pd.DataFrame(columns=feature_encoders.keys())
    for col in feature_encoders.keys():
        if col in features:
            input_features.at[0, col] = features[col]
        else:
            input_features.at[0, col] = "unknown"
    
    # Transform input features using the same encoders as during training
    encoded_input = pd.DataFrame()
    for feature, encoder in feature_encoders.items():
        feature_encoded = encoder.transform(input_features[feature].astype(str))
        feature_names = [f"{feature}_{x}" for x in encoder.get_feature_names_out()]
        encoded_df = pd.DataFrame.sparse.from_spmatrix(feature_encoded, columns=feature_names, index=input_features.index)
        encoded_input = pd.concat([encoded_input, encoded_df], axis=1)
    
    # Ensure all columns from training are present (fill missing with zeros)
    missing_cols = set(model.feature_names_in_) - set(encoded_input.columns)
    for col in missing_cols:
        encoded_input[col] = 0
    
    # Ensure columns are in the same order as during training
    encoded_input = encoded_input[model.feature_names_in_]
    
    # Get prediction and confidence scores
    prediction = model.predict(encoded_input)[0]
    
    # Get probability/confidence scores
    proba = model.predict_proba(encoded_input)[0]
    confidence = proba[prediction]
    
    # Get the predicted category name
    predicted_category = style_encoder.inverse_transform([prediction])[0]
    
    print(f"\nQuery: {query}")
    print(f"Extracted features: {features}")
    print(f"Recommended fashion category: {predicted_category.upper()}")
    print(f"Confidence score: {confidence:.2f}")
    
    # Generate additional suggestions based on the query
    # Find examples from the dataset that match the predicted category
    matching_rows = original_df[original_df['fashion_category'] == predicted_category]
    
    if not matching_rows.empty:
        item_type = features['item'] if features['item'] != 'unknown' else 'outfit'
        example = matching_rows.iloc[0]
        print(f"\nSuggested {item_type}: {example['AnswerText']}")
        
    return predicted_category, confidence

def main():
    """
    Main function to run the fashion recommendation system
    """
    file_path = "C:\\Users\\saabd\\OneDrive\\Desktop\\PRO\\n\\fashion_dataset.csv"
    
    # Load and inspect data
    df = load_and_inspect_data(file_path)
    
    # Preprocess data
    X, y, category_names, style_encoder, feature_encoders, original_df = preprocess_data(df)
    
    # Train and evaluate models
    best_model, results = train_and_evaluate_models(X, y, category_names.unique())
    
    print("\n=== Fashion Recommendation System Ready ===")
    print("You can now use the recommend_fashion() function to get recommendations!")
    
    # Example calls
    print("\n=== Example Recommendations ===")
    example_queries = [
        "What should I wear for a formal event in winter?",
        "I'm looking for casual summer outfit with blue colors",
        "Recommend a date night outfit for spring with something in red"
    ]
    
    for query in example_queries:
        recommend_fashion(query, best_model, style_encoder, feature_encoders, original_df)
        print("--------------------------------------------")
    
    return best_model, style_encoder, feature_encoders, original_df

if __name__ == "__main__":
    model, style_encoder, feature_encoders, df = main()
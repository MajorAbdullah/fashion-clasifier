import pandas as pd
import numpy as np
import re
import os
import sys
from fashion_classifier import load_and_inspect_data, preprocess_data, train_and_evaluate_models, recommend_fashion

class FashionRecommender:
    def __init__(self, dataset_path):
        """Initialize the fashion recommender system"""
        print("Loading fashion recommendation system...")
        # Load the dataset
        self.df = load_and_inspect_data(dataset_path)
        
        # Preprocess data
        self.X, self.y, self.category_names, self.style_encoder, self.feature_encoders, self.original_df = preprocess_data(self.df)
        
        # Train the model
        self.model, _ = train_and_evaluate_models(self.X, self.y, self.category_names.unique())
        
        print("\nFashion Recommender System initialized successfully!")
    
    def parse_clothing_items(self, row):
        """Extract clothing item from a dataset row"""
        item = None
        answer = row['AnswerText']
        tags = row['Tags'].split(',')
        
        if len(tags) > 0:
            item = tags[0]  # The first tag is typically the clothing item
        
        style = answer.split()[0]
        color = answer.split()[1] if len(answer.split()) > 1 else "unknown"
        material = answer.split()[2] if len(answer.split()) > 2 else "unknown"
        
        return {
            'item': item,
            'style': style,
            'color': color, 
            'material': material,
            'full_description': answer
        }
    
    def categorize_clothing(self, item):
        """Categorize clothing as top, bottom, or other"""
        tops = ['t-shirt', 'shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat']
        bottoms = ['pants', 'jeans', 'shorts', 'skirt', 'trousers']
        
        if item in tops:
            return 'top'
        elif item in bottoms:
            return 'bottom'
        else:
            return 'other'
    
    def ask_questions(self):
        """Ask the user a series of questions about their fashion preferences"""
        questions = [
            "What occasion are you dressing for? (e.g. casual, formal, work, party, date night, etc.)",
            "What season are you dressing for? (e.g. summer, winter, spring, autumn)",
            "Do you have any color preferences? (e.g. black, white, blue, red, etc.)",
            "Do you have any material preferences? (e.g. cotton, silk, wool, leather, etc.)",
            "What's your style preference? (e.g. casual, formal, punk, vintage, bohemian, etc.)"
        ]
        
        answers = {}
        for question in questions:
            print("\n" + question)
            answer = input("> ").strip().lower()
            
            # Store the answers by extracting the question type
            if "occasion" in question:
                answers["occasion"] = answer
            elif "season" in question:
                answers["season"] = answer
            elif "color" in question:
                answers["color"] = answer
            elif "material" in question:
                answers["material"] = answer
            elif "style" in question:
                answers["style"] = answer
        
        return answers
    
    def construct_query(self, answers):
        """Construct a query string from user answers"""
        query = f"Looking for {answers.get('style', 'casual')} outfit for {answers.get('occasion', 'casual')} in {answers.get('season', 'summer')}"
        
        if answers.get('color'):
            query += f" with {answers['color']} colors"
        
        if answers.get('material'):
            query += f" in {answers['material']} material"
        
        return query
    
    def recommend_outfit(self, answers):
        """Recommend a complete outfit based on user responses"""
        # Construct a query from user answers
        query = self.construct_query(answers)
        print(f"\nProcessing query: '{query}'")
        
        # Use the fashion_classifier to get the recommended style category
        predicted_category, confidence = recommend_fashion(
            query, self.model, self.style_encoder, self.feature_encoders, self.original_df
        )
        
        # Filter rows matching the predicted category
        matching_rows = self.original_df[self.original_df['fashion_category'] == predicted_category]
        
        # Extract all clothing items
        all_items = []
        for _, row in matching_rows.iterrows():
            item_info = self.parse_clothing_items(row)
            item_info['category'] = self.categorize_clothing(item_info['item'])
            all_items.append(item_info)
        
        # Separate into tops, bottoms, and others
        tops = [item for item in all_items if item['category'] == 'top']
        bottoms = [item for item in all_items if item['category'] == 'bottom']
        accessories = [item for item in all_items if item['category'] == 'other']
        
        # Select from tops and bottoms (prefer matching colors if provided)
        user_color = answers.get('color', '').lower()
        user_material = answers.get('material', '').lower()
        
        # Sort tops and bottoms by matching user preferences
        if user_color:
            tops = sorted(tops, key=lambda x: 1 if user_color in x['color'].lower() else 0, reverse=True)
            bottoms = sorted(bottoms, key=lambda x: 1 if user_color in x['color'].lower() else 0, reverse=True)
        
        if user_material:
            tops = sorted(tops, key=lambda x: 1 if user_material in x['material'].lower() else 0, reverse=True)
            bottoms = sorted(bottoms, key=lambda x: 1 if user_material in x['material'].lower() else 0, reverse=True)
        
        # Generate top 10 outfit combinations
        outfits = []
        for i in range(min(len(tops), 5)):
            for j in range(min(len(bottoms), 2)):
                top = tops[i]
                bottom = bottoms[j]
                outfit = {
                    'top': top,
                    'bottom': bottom,
                    'style': predicted_category
                }
                outfits.append(outfit)
                if len(outfits) >= 10:
                    break
            if len(outfits) >= 10:
                break
        
        # If we don't have 10 outfits yet and have some accessories, add them to outfits
        while len(outfits) < 10 and len(accessories) > 0:
            # Clone an existing outfit and add an accessory
            if len(outfits) > 0:
                outfit = outfits[0].copy()
                outfit['accessory'] = accessories.pop(0) if accessories else None
                outfits.append(outfit)
        
        # Fill any remaining slots with variations
        while len(outfits) < 10 and (len(tops) > 0 or len(bottoms) > 0):
            top = tops[0] if tops else {'item': 'any top', 'full_description': 'any top'}
            bottom = bottoms[0] if bottoms else {'item': 'any bottom', 'full_description': 'any bottom'}
            outfit = {
                'top': top,
                'bottom': bottom,
                'style': predicted_category
            }
            outfits.append(outfit)
        
        return outfits, predicted_category, confidence
    
    def display_outfits(self, outfits, style_category, confidence):
        """Display the recommended outfits"""
        print(f"\n===== TOP 10 RECOMMENDED OUTFITS ({style_category.upper()} STYLE) =====")
        print(f"Confidence: {confidence:.2f}")
        print("=" * 60)
        
        for i, outfit in enumerate(outfits):
            print(f"\nOutfit {i+1}:")
            print(f"  Top: {outfit['top']['full_description']} ({outfit['top']['item']})")
            print(f"  Bottom: {outfit['bottom']['full_description']} ({outfit['bottom']['item']})")
            if 'accessory' in outfit and outfit['accessory']:
                print(f"  Accessory: {outfit['accessory']['full_description']} ({outfit['accessory']['item']})")
            print("-" * 40)
        
        return

def main():
    """Main function to run the fashion recommender"""
    dataset_path = "C:\\Users\\saabd\\OneDrive\\Desktop\\PRO\\n\\fashion_dataset.csv"
    
    # Make sure the dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    # Initialize the recommender
    recommender = FashionRecommender(dataset_path)
    
    while True:
        print("\n\n====== FASHION OUTFIT RECOMMENDER ======")
        print("Let's find your perfect outfit! I'll ask you a few questions...")
        
        # Ask questions to get user preferences
        answers = recommender.ask_questions()
        
        # Get outfit recommendations
        outfits, style_category, confidence = recommender.recommend_outfit(answers)
        
        # Display recommendations
        recommender.display_outfits(outfits, style_category, confidence)
        
        # Ask if the user wants to continue
        print("\nWould you like to get more outfit recommendations? (yes/no)")
        continue_response = input("> ").strip().lower()
        if continue_response != 'yes' and continue_response != 'y':
            print("Thank you for using the Fashion Outfit Recommender! Goodbye!")
            break

if __name__ == "__main__":
    main()
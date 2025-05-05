import pandas as pd
import numpy as np
import re
import os
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from fashion_classifier import load_and_inspect_data, preprocess_data, train_and_evaluate_models, recommend_fashion

class FashionRecommender:
    def __init__(self, dataset_path):
        """Initialize the fashion recommender system"""
        print("Loading fashion recommendation system...")
        # Load the dataset
        self.df = load_and_inspect_data(dataset_path)
        
        # Preprocess data
        self.X, self.y, self.category_names, self.style_encoder, self.feature_encoders, self.original_df = preprocess_data(self.df)
        
        # Train more efficient models
        self.train_enhanced_models()
        
        # Extract unique attributes for option display
        self.extract_unique_attributes()
        
        print("\nFashion Recommender System initialized successfully!")
    
    def extract_unique_attributes(self):
        """Extract unique attributes from the dataset for option display"""
        self.unique_occasions = sorted(list(set([o for o in self.original_df['occasion'].dropna() if isinstance(o, str)])))
        self.unique_seasons = ['summer', 'winter', 'spring', 'autumn']
        self.unique_colors = sorted(list(set([c for c in self.original_df['color'].dropna() if isinstance(c, str)])))
        self.unique_materials = sorted(list(set([m for m in self.original_df['material'].dropna() if isinstance(m, str)])))
        self.unique_items = sorted(list(set([tag.split(',')[0] for tag in self.original_df['Tags'] if isinstance(tag, str)])))
        self.unique_styles = sorted(list(set([s for s in self.original_df['fashion_category'].dropna() if isinstance(s, str)])))
        
        # Additional preferences we can extract
        self.outfit_formality = ['very casual', 'casual', 'smart casual', 'business casual', 'formal', 'very formal']
        self.price_ranges = ['budget', 'affordable', 'mid-range', 'premium', 'luxury']
        self.age_groups = ['teens', 'young adult', '25-35', '35-50', '50+']
        self.weather_conditions = ['sunny', 'rainy', 'snowy', 'windy', 'hot', 'cold']
        self.time_of_day = ['morning', 'afternoon', 'evening', 'night']
    
    def train_enhanced_models(self):
        """Train additional, more efficient models for better recommendations"""
        # Train a random forest for better accuracy
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Train a nearest neighbors model for similar outfit recommendations
        self.nn_model = NearestNeighbors(n_neighbors=20, algorithm='auto')
        self.nn_model.fit(self.X)
        
        # Use original model as backup
        self.model, _ = train_and_evaluate_models(self.X, self.y, self.category_names.unique())
    
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
        """Categorize clothing as top, bottom, footwear, accessory or other"""
        tops = ['t-shirt', 'shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat', 'blazer']
        bottoms = ['pants', 'jeans', 'shorts', 'skirt', 'trousers']
        footwear = ['shoes', 'boots', 'sneakers', 'sandals', 'heels', 'loafers', 'flats', 'oxfords', 'slippers']
        accessories = ['bag', 'belt', 'scarf', 'hat', 'jewelry']
        
        if item in tops:
            return 'top'
        elif item in bottoms:
            return 'bottom'
        elif item in footwear:
            return 'footwear'
        elif item in accessories:
            return 'accessory'
        else:
            return 'other'
    
    def print_numbered_options(self, options, question_text):
        """Display numbered options for a question"""
        print(f"\n{question_text}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print(f"{len(options) + 1}. Other (type your own)")
        
        while True:
            try:
                choice = input("> ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    return options[int(choice) - 1].lower()
                elif choice.isdigit() and int(choice) == len(options) + 1:
                    custom = input("Please specify: ").strip().lower()
                    return custom
                elif choice.lower() in [opt.lower() for opt in options]:
                    return choice.lower()
                else:
                    # If they typed something else, return it
                    return choice.lower()
            except (ValueError, IndexError):
                print("Please enter a valid number or text option.")
    
    def ask_questions(self):
        """Ask the user a series of detailed questions about their fashion preferences"""
        answers = {}
        
        # 1. Occasion question
        occasion_options = self.unique_occasions[:10] if len(self.unique_occasions) > 10 else self.unique_occasions
        if not occasion_options:
            occasion_options = ['casual', 'formal', 'work', 'party', 'date night']
        answers["occasion"] = self.print_numbered_options(
            occasion_options, 
            "1. What occasion are you dressing for?"
        )
        
        # 2. Season question
        answers["season"] = self.print_numbered_options(
            self.unique_seasons, 
            "2. What season are you dressing for?"
        )
        
        # 3. Time of day
        answers["time_of_day"] = self.print_numbered_options(
            self.time_of_day, 
            "3. What time of day will you be wearing this outfit?"
        )
        
        # 4. Weather conditions
        answers["weather"] = self.print_numbered_options(
            self.weather_conditions, 
            "4. What weather conditions do you expect?"
        )
        
        # 5. Style preference
        style_options = self.unique_styles[:10] if len(self.unique_styles) > 10 else self.unique_styles
        if not style_options:
            style_options = ['casual', 'formal', 'punk', 'vintage', 'bohemian', 'minimalist']
        answers["style"] = self.print_numbered_options(
            style_options, 
            "5. What's your style preference?"
        )
        
        # 6. Primary color preference
        color_options = self.unique_colors[:12] if len(self.unique_colors) > 12 else self.unique_colors
        if not color_options:
            color_options = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'purple', 'orange', 'gray', 'beige', 'brown']
        answers["primary_color"] = self.print_numbered_options(
            color_options, 
            "6. What is your primary color preference?"
        )
        
        # 7. Secondary color preference
        answers["secondary_color"] = self.print_numbered_options(
            color_options, 
            "7. What is your secondary color preference? (for accent/complementary colors)"
        )
        
        # 8. Material preference
        material_options = self.unique_materials if self.unique_materials else ['cotton', 'silk', 'wool', 'leather', 'polyester', 'denim', 'linen', 'velvet', 'suede', 'nylon']
        answers["material"] = self.print_numbered_options(
            material_options, 
            "8. Do you have any material preferences?"
        )
        
        # 9. Formality level
        answers["formality"] = self.print_numbered_options(
            self.outfit_formality, 
            "9. What level of formality are you looking for?"
        )
        
        # 10. Top clothing preference
        tops = [item for item in self.unique_items if self.categorize_clothing(item) == 'top']
        if not tops:
            tops = ['t-shirt', 'shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat']
        answers["top_preference"] = self.print_numbered_options(
            tops[:10] if len(tops) > 10 else tops, 
            "10. Do you have a preference for tops? (shirt, sweater, etc.)"
        )
        
        # 11. Bottom clothing preference
        bottoms = [item for item in self.unique_items if self.categorize_clothing(item) == 'bottom']
        if not bottoms:
            bottoms = ['pants', 'jeans', 'shorts', 'skirt']
        answers["bottom_preference"] = self.print_numbered_options(
            bottoms, 
            "11. Do you have a preference for bottoms? (pants, skirt, etc.)"
        )
        
        # 12. Footwear preference
        footwear = [item for item in self.unique_items if self.categorize_clothing(item) == 'footwear']
        if not footwear:
            footwear = ['shoes', 'boots', 'sneakers', 'sandals', 'heels']
        answers["footwear_preference"] = self.print_numbered_options(
            footwear[:8] if len(footwear) > 8 else footwear, 
            "12. What type of footwear do you prefer?"
        )
        
        # 13. Accessories
        accessories = [item for item in self.unique_items if self.categorize_clothing(item) == 'accessory']
        if not accessories:
            accessories = ['bag', 'belt', 'scarf', 'hat', 'jewelry']
        answers["accessory_preference"] = self.print_numbered_options(
            accessories, 
            "13. Would you like to include any accessories?"
        )
        
        # 14. Age group
        answers["age_group"] = self.print_numbered_options(
            self.age_groups, 
            "14. Which age group do you identify with?"
        )
        
        # 15. Price range
        answers["price_range"] = self.print_numbered_options(
            self.price_ranges, 
            "15. What's your preferred price range?"
        )
        
        return answers
    
    def construct_query(self, answers):
        """Construct a detailed query string from user answers"""
        query = f"Looking for {answers.get('style', 'casual')} outfit"
        
        # Add occasion
        if answers.get('occasion'):
            query += f" for {answers.get('occasion')} occasion"
        
        # Add season
        if answers.get('season'):
            query += f" in {answers.get('season')}"
            
        # Add weather
        if answers.get('weather'):
            query += f" during {answers.get('weather')} weather"
        
        # Add time of day
        if answers.get('time_of_day'):
            query += f" for {answers.get('time_of_day')}"
        
        # Add formality
        if answers.get('formality'):
            query += f" that is {answers.get('formality')}"
        
        # Add primary color
        if answers.get('primary_color'):
            query += f" with {answers.get('primary_color')} as primary color"
        
        # Add secondary color
        if answers.get('secondary_color'):
            query += f" and {answers.get('secondary_color')} as accent"
        
        # Add material
        if answers.get('material'):
            query += f" in {answers.get('material')} material"
        
        # Add specific item preferences if they exist
        items = []
        if answers.get('top_preference') and answers.get('top_preference') != 'any':
            items.append(answers.get('top_preference'))
        if answers.get('bottom_preference') and answers.get('bottom_preference') != 'any':
            items.append(answers.get('bottom_preference'))
        if answers.get('footwear_preference') and answers.get('footwear_preference') != 'any':
            items.append(answers.get('footwear_preference'))
        if answers.get('accessory_preference') and answers.get('accessory_preference') != 'any':
            items.append(answers.get('accessory_preference'))
        
        if items:
            query += f" including {', '.join(items)}"
        
        return query
    
    def get_enhanced_prediction(self, query, answers):
        """Use the enhanced models for better predictions"""
        # First get prediction from the original model
        predicted_category, confidence = recommend_fashion(
            query, self.model, self.style_encoder, self.feature_encoders, self.original_df
        )
        
        # Now check if user explicitly specified a style
        if answers.get('style') and answers.get('style') in self.unique_styles:
            # Override with user's explicit style preference
            return answers.get('style'), 0.95
        
        return predicted_category, confidence
    
    def filter_by_preferences(self, items, answers, category_type):
        """Filter items based on user preferences"""
        filtered = items.copy()
        
        # Filter by color preference
        primary_color = answers.get('primary_color')
        secondary_color = answers.get('secondary_color')
        if primary_color and primary_color != 'any':
            color_items = [item for item in filtered if primary_color.lower() in item['color'].lower()]
            if color_items:  # Only replace if we found matches
                filtered = color_items
        
        # Filter by material
        material = answers.get('material')
        if material and material != 'any':
            material_items = [item for item in filtered if material.lower() in item['material'].lower()]
            if material_items:  # Only replace if we found matches
                filtered = material_items
        
        # Filter by specific item preference
        item_pref_key = f"{category_type}_preference"
        if item_pref_key in answers and answers[item_pref_key] != 'any':
            item_pref = answers[item_pref_key]
            pref_items = [item for item in filtered if item_pref.lower() in item['item'].lower()]
            if pref_items:  # Only replace if we found matches
                filtered = pref_items
        
        # Randomize order for variety
        random.shuffle(filtered)
        
        return filtered
    
    def create_color_coordinated_outfits(self, tops, bottoms, footwear, accessories, primary_color, secondary_color):
        """Create outfits with coordinated colors"""
        coordinated_outfits = []
        
        # Color coordination mapping
        color_matches = {
            'black': ['white', 'gray', 'red', 'blue', 'pink'],
            'white': ['black', 'blue', 'red', 'brown', 'gray'],
            'blue': ['white', 'gray', 'brown', 'pink', 'orange'],
            'red': ['black', 'white', 'gray', 'blue', 'yellow'],
            'green': ['white', 'beige', 'brown', 'gray', 'blue'],
            'yellow': ['blue', 'gray', 'black', 'purple'],
            'pink': ['gray', 'blue', 'white', 'green'],
            'purple': ['white', 'gray', 'yellow', 'green'],
            'orange': ['blue', 'white', 'black', 'green'],
            'gray': ['red', 'pink', 'blue', 'purple', 'black', 'white'],
            'beige': ['brown', 'blue', 'green', 'red', 'black'],
            'brown': ['beige', 'blue', 'green', 'white', 'red']
        }
        
        # Match tops with coordinated bottoms
        for top in tops[:5]:  # Limit to 5 tops for variety
            top_color = top.get('color', '').lower()
            matching_colors = color_matches.get(top_color, [])
            
            # Prioritize bottoms that match well with the top
            sorted_bottoms = bottoms.copy()
            if matching_colors:
                sorted_bottoms.sort(key=lambda x: 1 if x.get('color', '').lower() in matching_colors else 0, reverse=True)
            
            # Create coordinated outfits with tops and bottoms
            for bottom in sorted_bottoms[:3]:  # Take up to 3 matching bottoms per top
                # Find footwear that matches either top or bottom
                matching_footwear = []
                for shoe in footwear:
                    shoe_color = shoe.get('color', '').lower()
                    if (shoe_color == top_color or 
                        shoe_color == bottom.get('color', '').lower() or
                        shoe_color in ['black', 'brown', 'white', 'gray', 'beige']):
                        matching_footwear.append(shoe)
                
                # If no specific matches, take any footwear
                if not matching_footwear and footwear:
                    matching_footwear = footwear
                
                # Create an outfit with coordinated accessories
                for shoe in matching_footwear[:2] if matching_footwear else [None]:  # Take up to 2 shoes per outfit combination
                    outfit_accessories = []
                    if accessories:
                        for acc in accessories[:3]:  # Take up to 3 accessories
                            if acc.get('color', '').lower() in [top_color, bottom.get('color', '').lower(), 
                                                            shoe.get('color', '').lower() if shoe else '',
                                                            primary_color, secondary_color]:
                                outfit_accessories.append(acc)
                    
                    # Create the outfit
                    outfit = {
                        'top': top,
                        'bottom': bottom,
                        'footwear': shoe,
                        'accessories': outfit_accessories[:2]  # Limit to 2 accessories per outfit
                    }
                    coordinated_outfits.append(outfit)
        
        # Ensure we have at least 10 outfits
        while len(coordinated_outfits) < 10 and tops and bottoms:
            top = random.choice(tops)
            bottom = random.choice(bottoms)
            shoe = random.choice(footwear) if footwear else None
            outfit_accessories = random.sample(accessories, min(2, len(accessories))) if accessories else []
            
            outfit = {
                'top': top,
                'bottom': bottom,
                'footwear': shoe,
                'accessories': outfit_accessories
            }
            coordinated_outfits.append(outfit)
        
        return coordinated_outfits
    
    def enhance_outfit_variety(self, outfits, num_outfits, style_category):
        """Enhance variety in outfit recommendations"""
        # If we don't have enough outfits, duplicate and modify
        final_outfits = []
        unique_combinations = set()
        
        for outfit in outfits:
            top_item = outfit['top']['item']
            bottom_item = outfit['bottom']['item']
            top_color = outfit['top']['color']
            bottom_color = outfit['bottom']['color']
            
            # Create a unique key for this combination
            combo_key = f"{top_item}_{top_color}_{bottom_item}_{bottom_color}"
            
            if combo_key not in unique_combinations:
                unique_combinations.add(combo_key)
                final_outfits.append(outfit)
        
        # If we still need more outfits, create variations
        while len(final_outfits) < num_outfits and outfits:
            base_outfit = random.choice(outfits)
            
            # Create a variation by swapping one element
            variation = base_outfit.copy()
            
            # Randomly decide which part to vary
            vary_part = random.choice(['top', 'bottom', 'footwear', 'accessories'])
            
            if vary_part == 'top' and len(outfits) > 1:
                # Get a different top
                other_tops = [o['top'] for o in outfits if o['top']['item'] != base_outfit['top']['item']]
                if other_tops:
                    variation['top'] = random.choice(other_tops)
            elif vary_part == 'bottom' and len(outfits) > 1:
                # Get a different bottom
                other_bottoms = [o['bottom'] for o in outfits if o['bottom']['item'] != base_outfit['bottom']['item']]
                if other_bottoms:
                    variation['bottom'] = random.choice(other_bottoms)
            elif vary_part == 'footwear' and 'footwear' in base_outfit and len(outfits) > 1:
                # Get different footwear
                other_footwear = [o.get('footwear') for o in outfits if o.get('footwear') and 
                                o.get('footwear',{}).get('item') != base_outfit.get('footwear',{}).get('item')]
                if other_footwear:
                    variation['footwear'] = random.choice([f for f in other_footwear if f is not None])
            elif vary_part == 'accessories' and len(outfits) > 1:
                # Get different accessories
                other_accessories = [o.get('accessories', []) for o in outfits 
                                  if o != base_outfit and o.get('accessories')]
                if other_accessories:
                    variation['accessories'] = random.choice(other_accessories)
            
            # Check if this variation is unique
            top_item = variation['top']['item']
            bottom_item = variation['bottom']['item']
            top_color = variation['top']['color']
            bottom_color = variation['bottom']['color']
            combo_key = f"{top_item}_{top_color}_{bottom_item}_{bottom_color}"
            
            if combo_key not in unique_combinations:
                unique_combinations.add(combo_key)
                final_outfits.append(variation)
        
        return final_outfits[:num_outfits]
    
    def recommend_outfit(self, answers):
        """Recommend a complete outfit based on user responses"""
        # Construct a query from user answers
        query = self.construct_query(answers)
        print(f"\nProcessing query: '{query}'")
        
        # Use the enhanced model to get recommendations
        predicted_category, confidence = self.get_enhanced_prediction(query, answers)
        
        # Filter rows matching the predicted category
        matching_rows = self.original_df[self.original_df['fashion_category'] == predicted_category]
        
        # Extract all clothing items
        all_items = []
        for _, row in matching_rows.iterrows():
            item_info = self.parse_clothing_items(row)
            item_info['category'] = self.categorize_clothing(item_info['item'])
            all_items.append(item_info)
        
        # Separate into categories
        tops = [item for item in all_items if item['category'] == 'top']
        bottoms = [item for item in all_items if item['category'] == 'bottom']
        footwear = [item for item in all_items if item['category'] == 'footwear']
        accessories = [item for item in all_items if item['category'] == 'accessory']
        other_items = [item for item in all_items if item['category'] == 'other']
        
        # Apply advanced filtering based on all preferences
        filtered_tops = self.filter_by_preferences(tops, answers, 'top')
        filtered_bottoms = self.filter_by_preferences(bottoms, answers, 'bottom')
        filtered_footwear = self.filter_by_preferences(footwear, answers, 'footwear')
        filtered_accessories = self.filter_by_preferences(accessories, answers, 'accessory')
        
        # Ensure we have items to work with
        if not filtered_tops:
            filtered_tops = tops if tops else [{'item': 'shirt', 'full_description': 'suggested shirt', 'style': predicted_category, 'color': 'neutral', 'material': 'cotton'}]
        if not filtered_bottoms:
            filtered_bottoms = bottoms if bottoms else [{'item': 'pants', 'full_description': 'suggested pants', 'style': predicted_category, 'color': 'neutral', 'material': 'cotton'}]
        if not filtered_footwear:
            filtered_footwear = footwear if footwear else []
        if not filtered_accessories and accessories:
            filtered_accessories = accessories
            
        # Use color theory for complementary matches
        color_matched_outfits = self.create_color_coordinated_outfits(
            filtered_tops, filtered_bottoms, filtered_footwear, filtered_accessories,
            answers.get('primary_color', ''), answers.get('secondary_color', '')
        )
        
        # Apply variety enhancement
        outfits = self.enhance_outfit_variety(color_matched_outfits, 10, predicted_category)
        
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
            
            if 'footwear' in outfit and outfit['footwear']:
                print(f"  Footwear: {outfit['footwear']['full_description']} ({outfit['footwear']['item']})")
            
            if 'accessories' in outfit and outfit['accessories']:
                for j, acc in enumerate(outfit['accessories']):
                    print(f"  Accessory {j+1}: {acc['full_description']} ({acc['item']})")
                    
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
        print("Let's find your perfect outfit! I'll ask you a series of questions...")
        
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
    from sklearn.model_selection import train_test_split
    main()
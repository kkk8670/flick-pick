import argparse
import os
from src import data_processor, recommendation_engine
from web.app import app as flask_app

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--mode", choices=["preprocess", "train", "recommend", "serve", "all"], 
                        default="serve", help="Operation mode")
    parser.add_argument("--user_id", type=str, help="Specify user ID for recommendation")
    args = parser.parse_args()
    
    if args.mode in ["preprocess", "all"]:
        print("Preprocessing data...")
        data_processor.preprocess()
    
    if args.mode in ["train", "all"]:
        print("Training models...")
        recommendation_engine.train()
    
    if args.mode in ["recommend", "all"]:
        print("Generating recommendations...")
        if args.user_id:
            recommendation_engine.recommend_for_user(args.user_id)
        else:
            recommendation_engine.recommend_for_all_users()
    
    if args.mode in ["serve", "all"]:
        print("Starting web server...")
        flask_app.run(debug=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/23


import os, sys, json, logging, base64, traceback 
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from dotenv import load_dotenv
import plotly
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO
from threading import Thread

load_dotenv()
project_path = os.getenv("PYTHONPATH")
if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)
from stream.stream_recommend_engine import (
    init_spark,
    load_model_and_data,
    start_streaming,
    get_realtime_recommendations,
    get_latest_recommendations_from_csv
)


app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app.logger.setLevel(logging.INFO)


# Configuration
 
ROOT_DIR = os.getenv('ROOT_DIR')
RESULT_DIR = f"{ROOT_DIR}/data/output"
WORDCLOUD_DATA = f"{RESULT_DIR}/wordcloud_text.csv" 
NETWORK_DATA = f"{RESULT_DIR}/movie_similarity_network.csv"   
MOVIES_DATA = f"{RESULT_DIR}/wordcloud_text.csv"  # f"{RESULT_DIR}/movie_infos.csv" 
REALTIME_DATA = f"{RESULT_DIR}/realtime_streaming_recommend.csv"

ratings_df = pd.read_csv(f"{ROOT_DIR}/data/raw/ml-latest-small/ratings.csv")
tags_df = pd.read_csv(f"{ROOT_DIR}/data/raw/ml-latest-small/tags.csv")

last_read_time = None
recommendations_cache = {}


def format_ts(ts):
    try:
        return pd.to_datetime(ts, unit='s').strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def get_movie_info(movie_id):
    """Get movie information by movie ID"""
     
    movie_df = pd.read_csv(MOVIES_DATA)

    movie = movie_df[movie_df['movieId'] == movie_id].to_dict(orient='records')
    return movie[0] if movie else {}


def get_available_users():
    """Get all available user IDs"""
    users = []
    USER_DIR = f"{RESULT_DIR}/user_recommend_result"
    if os.path.exists(USER_DIR):
        for filename in os.listdir(USER_DIR):
            # app.logger.info(f"filename: {filename}")
            if filename.startswith("user_") and filename.endswith("_recommendation.csv"):
                # app.logger.info(f"choosen filename: {filename}")
                user_id = filename.split("_")[1]
                users.append(user_id)
    return users


def get_user_stats(user_id):
    user_id = int(user_id)
    stats = {
        "rated_movies": 0,
        "max_rating": None,
        "max_rating_movie": None,
        "max_rating_time": None,
        "tagged_movies": 0,
        "last_tag_movie": None,
        "last_tag": None,
        "last_tag_time": None,
    }
    try:
        movie_df = pd.read_csv(MOVIES_DATA)
        user_rating_df = ratings_df[ratings_df["userId"] == user_id]
        if not user_rating_df.empty:
            stats["rated_movies"] = user_rating_df["movieId"].nunique()
            max_rating = user_rating_df["rating"].max()
            stats["max_rating"] = float(max_rating)
            top_rated = user_rating_df[user_rating_df["rating"] == max_rating]
            top_row = top_rated.loc[top_rated["timestamp"].idxmax()]
            stats["max_rating_time"] = format_ts(top_row["timestamp"])
            movie_title_row = movie_df[movie_df["movieId"] == top_row["movieId"]]["title"]
            if not movie_title_row.empty:
                stats["max_rating_movie"] = movie_title_row.values[0]

        user_tag_df = tags_df[tags_df['userId'] == user_id]
        if not user_tag_df.empty:
            stats["tagged_movies"] = user_tag_df["movieId"].nunique()
            # print("tagged_movies", stats["tagged_movies"])
            last_tag = user_tag_df.loc[user_tag_df["timestamp"].idxmax()]
            stats["last_tag_time"] = format_ts(last_tag["timestamp"])
            stats["last_tag"] = last_tag["tag"]
            last_tag_title = movie_df[movie_df["movieId"] == last_tag["movieId"]]["title"]
            if not last_tag_title.empty:
                stats["last_tag_movie"] = last_tag_title.values[0]
        return stats

    except Exception as e:
        print(f"Error computing user stats for user {user_id}: {e}")
        return {}


def get_user_recommendations(user_id):
    """Get recommendations for a specific user"""
    file_path = f"{RESULT_DIR}/user_recommend_result/user_{user_id}_recommendation.csv" 
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Add movie title and other information
        recommendations = []
        for _, row in df.iterrows():
            movie_info = get_movie_info(row['movieId'])
            recommendations.append({
                'movieId': row['movieId'],
                'rating': row['rating'],
                'similarity': row['similarity'],
                'final_score': row['final_score'],
                'title': movie_info['title'],
                # 'genres': movie_info['genres']
            })
        return recommendations
    return []


def generate_wordcloud_image():
    """Generate wordcloud image as base64 string"""
    if os.access(WORDCLOUD_DATA, os.R_OK):
        try:
            df = pd.read_csv(WORDCLOUD_DATA)

            words = " ".join(df['wordcloud_text'].dropna().astype(str)).split()
            words = [w.lower() for w in words if w.isalpha()]  

            words = [w for w in words if 2 <= len(w) <= 15]
    
            freq_dict = pd.Series(words).value_counts().to_dict()
 
            stopwords = set(STOPWORDS)
            stopwords.update({'https', 'http', 'com', 'www', 'the', 'in'})   

            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color='white',
                max_words=200,
                prefer_horizontal=0.6,
                scale=3,
                relative_scaling=0.5,
                max_font_size=120,
                min_font_size=12,
                stopwords=stopwords,
                collocations=False,   
                colormap="viridis",
                normalize_plurals=True, 
            ).generate_from_frequencies(freq_dict)

            img = BytesIO()
            wordcloud.to_image().save(img, format='PNG')
            img.seek(0)
            return base64.b64encode(img.getvalue()).decode()
        except Exception as e:
            print(f"Error generating wordcloud: {e}")
    return None


def adjust_positions_for_overlap(pos, min_distance=0.2):
    nodes = list(pos.keys())
    adjusted = True
    max_iterations = 50  
    iteration = 0
    
    while adjusted and iteration < max_iterations:
        adjusted = False
        iteration += 1
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
         
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                dx = x2 - x1
                dy = y2 - y1
                distance = (dx**2 + dy**2)**0.5
            
                if distance < min_distance:
                    adjusted = True
                    
                    
                    if distance > 0:
                        move_x = dx / distance * (min_distance - distance) / 2
                        move_y = dy / distance * (min_distance - distance) / 2
                    else:  # totally overlap
                        move_x = min_distance / 2
                        move_y = 0
                    
                    # move nodes offset
                    pos[node1] = (x1 - move_x, y1 - move_y)
                    pos[node2] = (x2 + move_x, y2 + move_y)
    return pos


def generate_network_graph():
    """Generate movie similarity network graph as JSON"""
    # app.logger.info(f"func if file exists at: {NETWORK_DATA}")
    if os.access(NETWORK_DATA, os.R_OK):
        try:
            df = pd.read_csv(NETWORK_DATA)
            
            # Create network graph
            G = nx.Graph()
            
            # Assume CSV has source, target and weight columns
            for _, row in df.iterrows():
                source = row['sourceMovie']
                target = row['targetMovie']
                weight = row['similarity']
                # filter
                if weight > 0.6:  
                    G.add_edge(source, target, weight=weight)
            
            # group by Community detection algorithm
            communities = nx.community.greedy_modularity_communities(G)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Force Oriented Layout
            # adjust edge lenght
            pos = nx.spring_layout(G, k=0.8, iterations=200, weight='weight')
            pos = adjust_positions_for_overlap(pos)
          
            edge_traces = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = edge[2]['weight']
                
                # add width
                width = weight * 5
                
                # add color
                color = f"rgba(65, 105, 225, {weight})"   
                
                # add hover label
                hover_text = f"similarity: {weight:.2f}"
                
                # add trace for each edge
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color=color),
                    hoverinfo='text',
                    text=hover_text,
                    mode='lines',
                    showlegend=False
                )
                
                edge_traces.append(edge_trace)
            
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_texts = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # for node 
                # 1. connect number
                neighbors = list(G.neighbors(node))
                adjacencies = len(neighbors)
                
                # 2. sum weight
                weight_sum = 0
                for neighbor in neighbors:
                    weight_sum += G[node][neighbor]['weight']
                
                # adjust node size
                node_size = 10 + (adjacencies * 5) + (weight_sum * 10)
                node_sizes.append(node_size)
                
                # map color based on community
                community_id = community_map.get(node, 0)
                node_colors.append(community_id)
                
                movie_info = get_movie_info(node)
                title = movie_info.get('title', 'Unknown')
                genre = movie_info.get('genre', '')
                year = movie_info.get('year', '')
                
              
                node_texts.append(f"<b>{title}</b> ({year})<br>type: {genre}<br>conn num: {adjacencies}<br>similarity sum: {weight_sum:.2f}")
            
            # create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_texts,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=node_sizes,
                    color=node_colors,
                    colorbar=dict(
                        title='moive similarity',
                        thickness=15,
                        xanchor='left'
                    ),
                    line=dict(width=1, color='#333')
                )
            )
            
            # add labels
            labels = []
            label_x = []
            label_y = []
            
            # cal score of all node
            importance_scores = {}
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                adjacencies = len(neighbors)
                
                weight_sum = 0
                for neighbor in neighbors:
                    weight_sum += G[node][neighbor]['weight']
                
                # score = connect number + weight_sum * 2
                importance_scores[node] = adjacencies + weight_sum * 2
            
            # Identify the top 30% most important nodes
            threshold = sorted(importance_scores.values(), reverse=True)[int(len(importance_scores) * 0.3)] if importance_scores else 0
            
            # add label for important nodes
            for node, score in importance_scores.items():
                if score >= threshold:  # filter important node
                    movie_info = get_movie_info(node)
                    title = movie_info.get('title', 'Unknown')
                    x, y = pos[node]
                    labels.append(title)
                    label_x.append(x)
                    label_y.append(y)
            
            label_y_offset = []
            for y in label_y:
                label_y_offset.append(y + 0.2)

            label_trace = go.Scatter(
                x=label_x,
                y=label_y_offset,
                mode='text',
                text=labels,
                textposition="top center",
                textfont=dict(
                    family="Arial",
                    size=10,
                    color="black"
                )
            )
            
            # create diagram
            fig = go.Figure()
            
            # add edge trace
            for edge_trace in edge_traces:
                fig.add_trace(edge_trace)
                
            fig.add_trace(node_trace)
            fig.add_trace(label_trace)
            
            fig.update_layout(
                title=dict(
                    text='Movie similarity network diagram',
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,  
                width=900,   
                plot_bgcolor='#f8f9fa'  
            )
                
            annotations = [
                dict(
                    x=0.01,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    # text="Node size: number of connections + sum of similarities<br>Node color: movie community<br>Edge width and color depth: similarity",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="#ffffff",
                    bordercolor="#333333",
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )
            ]
            fig.update_layout(annotations=annotations)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            app.logger.error(f"Error generating network graph: {e}")
            app.logger.error(traceback.format_exc())
    return None


 

# @app.before_request
# def check_files():
#     app.logger.info("Flask server starting...")
#     files_to_check = [WORDCLOUD_DATA, NETWORK_DATA, MOVIES_DATA]
#     for file_path in files_to_check:
#         if not os.path.exists(file_path):
#             app.logger.error(f"File not found: {file_path}")
#         else:
#             app.logger.info(f"File exists: {file_path}")


@app.route('/test')
def test():
    app.logger.info("Test route accessed")
    users = get_available_users()
    return jsonify({
        "status": "ok",
        "file_exists": {
            "wordcloud": os.path.exists(WORDCLOUD_DATA),
            "network": os.path.exists(NETWORK_DATA),
            "movies": os.path.exists(MOVIES_DATA)
        },
        "users": users,
        "paths": {
            "root": ROOT_DIR,
            "result": RESULT_DIR
        } 
    })


@app.route('/')
def index():
    users = get_available_users()
    selected_user = request.args.get('user_id', users[0] if users else None)
    # app.logger.info(f"users: {users}, selected_user: {selected_user}")
    # Get recommendations for selected user
    recommendations = get_user_recommendations(selected_user) if selected_user else []
    # app.logger.info(f"index if file exists at: {NETWORK_DATA}")

    # Generate wordcloud
    wordcloud_img = generate_wordcloud_image()
    
    # Generate network graph
    graph_json = generate_network_graph()
    # app.logger.info(f"graph_json: {graph_json}")
    
    return render_template('dashboard.html', 
                          users=users,
                          selected_user=selected_user,
                          recommendations=recommendations,
                          wordcloud_img=wordcloud_img,
                          graph_json=graph_json)


@app.route('/debug-graph')
def debug_graph():
    graph_json = generate_network_graph()
    return f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            const graphData = {graph_json};
            console.log(graphData);
            Plotly.newPlot('graph', graphData.data, graphData.layout);
        </script>
    </body>
    </html>
    """

@app.route('/api/recommendations/<user_id>')
def api_recommendations(user_id):
    recommendations = get_user_recommendations(user_id)
    return jsonify(recommendations)


@app.route("/api/user_stats/<int:user_id>")
def api_user_stats(user_id):
    try:
        stats = get_user_stats(user_id)
        # app.logger.info(f"DEBUG STATS:: {stats}")
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

 
@app.route('/api/realtime_recommendations', methods=['GET'])
def api_realtime_recommendations():
    user_id = request.args.get('user_id', type=int)
    if user_id:
        recommendations = get_realtime_recommendations(user_id, 10, spark, model, movies)
        app.logger.info(f"num of rec 1111:: {len(recommendations)}")
    else: 
        recommendations = get_latest_recommendations_from_csv()
        app.logger.info(f"num of rec 2222:: {len(recommendations)}")
    return jsonify(recommendations)

 
@app.route('/api/realtime_user_recommendations/<int:user_id>', methods=['GET'])
def api_user_realtime_recommendations(user_id):
 
    recommendations = get_realtime_recommendations(user_id, 10, spark, model, movies)
    app.logger.info(f"num of rec 3333:: {len(recommendations)}")
    return jsonify(recommendations)

def start_background_streaming():
    global spark, model, movies
    spark = init_spark()
    model, movies = load_model_and_data(spark)
    query = start_streaming(spark, model, movies)
    query.awaitTermination()


if __name__ == '__main__':
    streaming_thread = Thread(target=start_background_streaming)
    streaming_thread.daemon = True 
    streaming_thread.start()

    app.run(host="127.0.0.1", port=5000, debug=True)
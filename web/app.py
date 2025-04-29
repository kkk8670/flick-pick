import os, sys, json, logging, base64, traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from dotenv import load_dotenv
import plotly
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
 


app = Flask(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app.logger.setLevel(logging.DEBUG)


# Configuration
load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
RESULT_DIR = f"{ROOT_DIR}/data/output"
WORDCLOUD_DATA = f"{RESULT_DIR}/wordcloud_text.csv" 
NETWORK_DATA = f"{RESULT_DIR}/movie_similarity_network.csv"   
MOVIES_DATA = f"{RESULT_DIR}/movie_infos.csv" 

movie_df = {}

def get_movie_info(movie_id):
    """Get movie information by movie ID"""
    if not movie_df and os.path.exists(MOVIES_DATA):
        movies_df = pd.read_csv(MOVIES_DATA)

    movie = movies_df[movies_df['movieId'] == movie_id].to_dict(orient='records')
    return movie[0] if movie else {}


def get_available_users():
    """Get all available user IDs"""
    users = []
    if os.path.exists(RESULT_DIR):
        for filename in os.listdir(RESULT_DIR):
            app.logger.info(f"filename: {filename}")
            if filename.startswith("user_") and filename.endswith("_recommendation.csv"):
                app.logger.info(f"choosen filename: {filename}")
                user_id = filename.split("_")[1]
                users.append(user_id)
    return users


def get_user_recommendations(user_id):
    """Get recommendations for a specific user"""
    file_path = f"{RESULT_DIR}/user_{user_id}_recommendation.csv" 
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
            print(df.head())
            # Assuming CSV has text and weight columns
            # text_data = " ".join([word * int(weight) for word, weight in zip(df['wordcloud_text'], df['weight'])])
            # wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
            
            # # Convert to base64 for HTML display
            # img = BytesIO()
            # wordcloud.to_image().save(img, format='PNG')
            # img.seek(0)
            # return base64.b64encode(img.getvalue()).decode()
        except Exception as e:
            print(f"Error generating wordcloud: {e}")
    return None

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
                G.add_edge(source, target, weight=weight)
            
            # Use Plotly to create interactive network graph
            pos = nx.spring_layout(G)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title=dict(text='Node Connections', side='right'),
                        xanchor='left'
                    )
                )
            )
            
            # Add node information
            node_adjacencies = []
            node_text = []
            for node in G.nodes():
                adjacencies = len(list(G.neighbors(node)))
                node_adjacencies.append(adjacencies)
                movie_info = get_movie_info(node)
                node_text.append(f"Movie ID: {node}<br>Title: {movie_info['title']}<br>Connections: {adjacencies}")
                
            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text
            
            fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='Movie Similarity Network',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

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
    app.logger.info(f"users: {users}")
    app.logger.info(f"selected_user: {selected_user}")
    # Get recommendations for selected user
    recommendations = get_user_recommendations(selected_user) if selected_user else []
    # app.logger.info(f"index if file exists at: {NETWORK_DATA}")
    # Generate wordcloud
    wordcloud_img = generate_wordcloud_image()
    
    # Generate network graph
    graph_json = generate_network_graph()
    
    return render_template('dashboard.html', 
                          users=users,
                          selected_user=selected_user,
                          recommendations=recommendations,
                          wordcloud_img=wordcloud_img,
                          graph_json=graph_json)


@app.route('/api/recommendations/<user_id>')
def api_recommendations(user_id):
    recommendations = get_user_recommendations(user_id)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
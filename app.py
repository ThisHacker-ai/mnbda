from collections import Counter, defaultdict
import random
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import requests
# from bs4 import BeautifulSoup
import ast
import re
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import database
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)


database.init_db()


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


def normalize_title(title):
    return re.sub(r'[^\w\s]', '', title).strip().lower()


def jaccard_genre_similarity(genres1, genres2):
    set1 = set([g.strip().lower() for g in genres1])
    set2 = set([g.strip().lower() for g in genres2])
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0


user_feedback = {
    "liked_genres": [],
    "disliked_genres": []
}

prev_similar_list_global = []


df = pd.read_csv(r'D:\pythoncode\recommender\tmdb_5000_movies.csv')

# df.drop(['release_date', 'homepage', 'original_title', 'production_companies', 'production_countries', 'revenue', 'spoken_languages'
# 'status', 'crew', 'budget'], axis=1, inplace=True)


if 'User_Feedback' not in df.columns:
    df['User_Feedback'] = None


if 'username' not in df.columns:
    df['username'] = None  # or "" if its an empty string


df['username'] = df['username'].fillna("")


df['keywords'] = df['keywords'].fillna('')
df['genres'] = df['genres'].fillna('')
df['director'] = df['director'].fillna('')
df['combined_text'] = df['keywords'] + \
    ' ' + df['genres'] + ' ' + df['director']

df['normalized_title'] = df['title'].apply(normalize_title)


df['vote_count'] = df['vote_count'].astype(
    str).str.replace(',', '').str.extract(r'(\d+)')

df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')

df = df[df['vote_count'] >= 200].reset_index(drop=True)

print(df.columns.to_list())

# User class for login manager


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


# title image scraper
'''def get_book_image_url(book_url):
    try:
        response = requests.get(book_url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', {'class': 'ResponsiveImage'})
        if img_tag:
            return img_tag['src']
    except:
        pass
    return '''''

book_url = "https://cdn-icons-png.flaticon.com/512/29/29302.png"


prevbooks = []


def parse_genres(genres_field):
    if not genres_field or not isinstance(genres_field, str):
        return []

    genres_field = genres_field.strip()

    try:
        # Try parsing list-like strings
        if genres_field.startswith('[') and genres_field.endswith(']'):
            genres = ast.literal_eval(genres_field)
            if all(isinstance(g, str) and len(g) > 1 for g in genres):
                return [g.strip() for g in genres]
    except:
        pass

    # Fallback to manual split
    if ',' in genres_field:
        genres = [g.strip() for g in genres_field.split(',')]
    else:
        genres = [g.strip() for g in genres_field.split()]

    return [g for g in genres if g]


def extract_genres(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith('[') else [value]
    except:
        return []


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m)) * R + (m / (m + v)) * C


def jaccard_genre_similarity(list1, list2):
    set1 = set([g.lower().strip() for g in list1])
    set2 = set([g.lower().strip() for g in list2])
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0


# Core recommendation logic
def search_book_details(search_word):
    search_lower = search_word.strip().lower()

    # Step 1: Genre-based recommendation
    genre_matched_books = pd.DataFrame()
    for idx, row in df.iterrows():
        genres = parse_genres(row.get('genres', ''))
        if any(search_lower == g.lower() for g in genres):
            genre_matched_books = pd.concat(
                [genre_matched_books, row.to_frame().T])

    if not genre_matched_books.empty:
        genre_matched_books = genre_matched_books.drop_duplicates(
            subset='title')
        genre_matched_books['vote_count'] = pd.to_numeric(
            genre_matched_books['vote_count'], errors='coerce')
        genre_matched_books['vote_average'] = pd.to_numeric(
            genre_matched_books['vote_average'], errors='coerce')

        # Filter based on vote count threshold
        filtered_books = genre_matched_books[genre_matched_books['vote_count'] >= 1500]
        if filtered_books.empty:
            filtered_books = genre_matched_books.copy()

        C = df['vote_average'].mean()
        m = 1500
        filtered_books['weighted_rating'] = filtered_books.apply(
            lambda x: weighted_rating(x, m, C), axis=1)
        top_books = filtered_books.sort_values(
            by='weighted_rating', ascending=False).drop_duplicates(subset='title').head(5)

        matched_book = {
            'title': f"Top Rated in Genre: {search_word.title()}",
            'desc': f"Showing top-rated books in the genre '{search_word}'.",
            'rating': '',
            'img': 'https://cdn-icons-png.flaticon.com/512/29/29302.png'
        }

        # Prepare combined text for similarity (genre + keywords + director)
        filtered_books['combined_text'] = filtered_books.apply(
            lambda row: (
                (row.get('genres', '') or '') + ' ' +
                (row.get('keywords', '') or '') + ' ' +
                (row.get('director', '') or '')
            ), axis=1)

        # Vectorize combined text for similarity scoring
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(
            filtered_books['combined_text'])
        query_vector = vectorizer.transform([search_word])
        cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

        filtered_books['similarity'] = cosine_sim
        similar_books = filtered_books.sort_values(
            by='similarity', ascending=False).head(5)

        similar_list = [{
            'title': row['title'],
            'desc': row.get('keywords', ''),
            'rating': row.get('vote_average', ''),
            'img': book_url
        } for _, row in similar_books.iterrows()]

        return matched_book, similar_list, [], []

    # Step 2: Direct match in title/author
    direct_matches = df[
        df['title'].str.lower().str.contains(search_lower, na=False) |
        df['director'].str.lower().str.contains(search_lower, na=False)
    ].drop_duplicates(subset='title')

    if not direct_matches.empty:
        reference_row = direct_matches.iloc[0]
    else:
        keyword_matches = df[
            df['keywords'].str.lower().str.contains(search_lower, na=False) |
            df['genres'].str.lower().str.contains(search_lower, na=False)
        ]
        if keyword_matches.empty:
            return None, [], [], []
        reference_row = keyword_matches.iloc[0]

    desc = reference_row.get('keywords', '') or ''
    genres = reference_row.get('genres', '') or desc
    genres_list = extract_genres(genres)
    search_theme = ' '.join(genres_list) + ' ' + desc + \
        ' ' + reference_row.get('director', '')

    reference_genres = extract_genres(reference_row.get('genres', ''))

    # Prepare text data for similarity scoring on full df
    df['combined_text'] = df.apply(lambda row: (
        (row.get('genres', '') or row.get('keywords', '')) + ' ' +
        (row.get('keywords', '') or row.get('genres', '')) + ' ' +
        row.get('director', '')
    ), axis=1)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    query_vector = vectorizer.transform([search_theme])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

    df['similarity'] = cosine_sim
    df['rating'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
    df['genre_similarity_jaccard'] = df['genres'].apply(extract_genres).apply(
        lambda genres: jaccard_genre_similarity(reference_genres, genres)
    )

    df['score'] = (
        df['similarity'] * 0.4 +
        df['genre_similarity_jaccard'] * 0.2 +
        df['popularity'].rank(pct=True) * 0.1 +
        np.log1p(df['vote_count']).rank(pct=True) * 0.1
    ) * df['vote_average']

    matched_book = {
        'title': reference_row['title'],
        'desc': reference_row.get('keywords', ''),
        'rating': reference_row.get('vote_average', ''),
        'img': book_url
    }

    author_name = reference_row.get('director', '').strip().lower()
    matched_title = reference_row['title']
    prevbooks.append(matched_title)

    df_filtered = df[
        (df['title'] != matched_title) &
        (df['director'].str.strip().str.lower() != author_name)
    ]

    similar_books = df_filtered.sort_values(
        by='score', ascending=False).head(5)
    similar_list = [{
        'title': row['title'],
        'desc': row.get('keywords', ''),
        'rating': row.get('vote_average', ''),
        'img': book_url
    } for _, row in similar_books.iterrows()]

    author_books = df[
        (df['director'].str.strip().str.lower() == author_name) &
        (df['title'] != matched_title)
    ]
    author_list = [{
        'title': row['title'],
        'desc': row.get('keywords', ''),
        'rating': row.get('vote_average', ''),
        'img': book_url
    } for _, row in author_books.iterrows()]

    return matched_book, similar_list, prev_similar_list_global, author_list


def add_initial_user(username, password, db_path='movies_app.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Generate hashed password
    password_hash = generate_password_hash(password)
    try:
        cursor.execute(
            'INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash)
        )
        conn.commit()
        print(f"User '{username}' added successfully!")
    except Exception as e:
        print(f"Error adding user: {e}")
    finally:
        conn.close()


# Flask routes

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    fresh = df[df['vote_average'] > 4.0].sample(n=5).to_dict(orient='records')
    matched_book, similar_books, prev_similar_list, author_books = None, [], [], []
    global prev_similar_list_global

    if request.method == 'POST':
        search_word = request.form['book_title']
        if search_word.strip():
            matched_book, similar_books, prev_similar_list, author_books = search_book_details(
                search_word)
            prev_similar_list_global = prev_similar_list

    return render_template('index.html',
                           matched_book=matched_book,
                           similar_books=similar_books,
                           prev_similar_list=prev_similar_list_global,
                           author_books=author_books,
                           fresh=fresh,
                           username=current_user.username)


@app.before_request
def check_user():
    # Check if user is not authenticated
    # Also check request.endpoint exists and is not 'login' or 'register'
    if not current_user.is_authenticated and \
       (request.endpoint is None or request.endpoint not in ['login', 'register']):
        # or redirect(url_for('register'))
        return render_template('register.html')


@login_manager.user_loader
def load_user(user_id):
    if user_id is None:
        return None
    conn = sqlite3.connect('movies_app.db')
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, username FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(id=row[0], username=row[1])
    return None


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        flash("Registered successfully! You can now log in.")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("movies_app.db")  # Use the correct DB
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, password_hash FROM users WHERE username=?", (
                username,)
        )
        row = cur.fetchone()
        conn.close()

        if row:
            user_id, user_name, user_password_hash = row
            if check_password_hash(user_password_hash, password):
                user = User(user_id, user_name)
                login_user(user)
                flash(f"Welcome, {user.username}!", "success")
                return redirect(url_for("index"))
            else:
                flash("Incorrect password. Please try again.", "danger")
        else:
            flash("User does not exist. Please register first.", "warning")

    return render_template("login.html")


@app.route('/dashboard')
@login_required
def dashboard():
    username = current_user.username
    user_id = database.add_user(username)

    # GET FEEDBACK DATA FROM DATABASE
    conn = sqlite3.connect('movies_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT book_id, feedback_time FROM feedback
        WHERE user_id = ? AND liked = 1
    ''', (user_id,))
    feedback_rows = cursor.fetchall()
    conn.close()

    # If no feedback, show empty state
    if not feedback_rows:
        return render_template('dashboard.html',
                               username=username,
                               most_liked_genre="Not enough data yet",
                               recent_book={},
                               recommended_books=[],
                               quote=random.choice([
                                   "Reading gives us someplace to go when we have to stay where we are.",
                                   "Until I feared I would lose it, I never loved to read. One does not love breathing.",
                                   "We lose ourselves in books, we find ourselves there too."
                               ]),
                               suggested_author="Not enough data")

    liked_book_ids = [row[0] for row in feedback_rows]

    # PROCESS LIKED BOOKS FROM FEEDBACK
    liked_books_df = df[df['title'].isin(liked_book_ids)]

    # Parse and normalize liked genres
    liked_genres = []
    for g in liked_books_df['genres']:
        try:
            genres = ast.literal_eval(g) if isinstance(
                g, str) and g.startswith('[') else [g]
            liked_genres.extend([x.strip().lower() for x in genres])
        except:
            continue

    # Now count genres with duplicates preserved
    genre_counts = Counter(liked_genres)
    most_liked_genre = max(
        genre_counts, key=genre_counts.get) if genre_counts else "Not enough data yet"

    # Most recently liked book
    most_recent_book_title = sorted(
        feedback_rows, key=lambda x: x[1], reverse=True)[0][0]
    recent_book = df[df['title'] == most_recent_book_title].iloc[0].to_dict(
    ) if most_recent_book_title in df['title'].values else {}

    # PREPROCESSING
    # Parse genres column as list for all books
    def parse_genres(val):
        if isinstance(val, str) and val.startswith('['):
            try:
                return [x.strip().lower() for x in ast.literal_eval(val)]
            except:
                return [val.lower()]
        else:
            return [val.lower()] if isinstance(val, str) else []

    df['genres_List'] = df['genres'].apply(parse_genres)

    # Filter out books the user already liked
    df_filtered = df[~df['title'].isin(liked_book_ids)].copy()

    # WEIGHTED RATING
    C = df['vote_average'].mean()
    if 'vote_count' in df.columns:
        m = df['vote_count'].quantile(0.75)
        df_filtered['vote_count'] = pd.to_numeric(
            df_filtered['vote_count'], errors='coerce').fillna(0)
        df_filtered['Weighted_Score'] = (
            (df_filtered['vote_count'] /
             (df_filtered['vote_count'] + m)) * df_filtered['vote_average']
            + (m / (df_filtered['vote_count'] + m)) * C
        )
    else:
        df_filtered['Weighted_Score'] = df_filtered['vote_average']

    # SCORE BOOKS BASED ON GENRE MATCHES
    def genre_boost(genres):
        # Count how many liked genres appear in this book's genres
        matches = sum([genre_counts.get(g, 0) for g in genres])
        return matches

    df_filtered['Genre_Boost'] = df_filtered['genres_List'].apply(genre_boost)

    # Final score = weighted rating + genre boost
    df_filtered['Final_Score'] = df_filtered['Weighted_Score'] + \
        df_filtered['Genre_Boost']

    # Sort books by final score descending
    df_filtered = df_filtered.sort_values('Final_Score', ascending=False)

    # Take top 10 unique books
    recommended_books = df_filtered.head(
        10)[['title', 'director', 'vote_average', 'genres']].to_dict(orient='records')

    # DATABASE LOGGING
    query_log = ','.join(genre_counts.elements()
                         ) if genre_counts else "dashboard_visit"
    database.add_search(user_id, query_log)
    rec_book_ids = [book['title'] for book in recommended_books]
    database.add_recommendations(
        user_id, rec_book_ids, rec_type='dashboard_genre_recommendation')

    # QUOTE & AUTHOR SUGGESTION
    quote = random.choice([
        "Reading gives us someplace to go when we have to stay where we are.",
        "Until I feared I would lose it, I never loved to read. One does not love breathing.",
        "We lose ourselves in books, we find ourselves there too."
    ])

    if most_liked_genre != "Not enough data yet":
        top_genre_books = df[df['genres_List'].apply(
            lambda x: most_liked_genre in x)]
        suggested_author = top_genre_books['director'].mode(
        ).iloc[0] if not top_genre_books.empty else "Unknown"
    else:
        suggested_author = "Not enough data"

    # COLLABORATIVE GENRE-BASED RECOMMENDATIONS
    conn = sqlite3.connect('movies_app.db')
    cursor = conn.cursor()

    cursor.execute('SELECT user_id, book_id FROM feedback WHERE liked = 1')
    all_likes = cursor.fetchall()

    user_genres = defaultdict(set)
    user_books = defaultdict(set)
    for uid, book_id in all_likes:
        book_row = df[df['title'] == book_id]
        if not book_row.empty:
            genres = book_row.iloc[0]['genres_List']
            user_genres[uid].update(genres)
            user_books[uid].add(book_id)
    conn.close()

    current_user_genres = user_genres.get(user_id, set())
    current_user_books = user_books.get(user_id, set())

    similar_users = [uid for uid, genres in user_genres.items(
    ) if uid != user_id and len(current_user_genres & genres) >= 4]

    similar_user_recs = set()
    for uid in similar_users:
        similar_user_recs.update(user_books[uid] - current_user_books)

    collab_books_df = df[df['title'].isin(similar_user_recs)][[
        'title', 'director', 'vote_average', 'genres']].drop_duplicates().head(10)
    collab_books = collab_books_df.to_dict(orient='records')

    return render_template(
        'dashboard.html',
        username=username,
        most_liked_genre=most_liked_genre,
        recent_book=recent_book,
        recommended_books=recommended_books,
        quote=quote,
        suggested_author=suggested_author,
        collab_books=collab_books
    )


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for('login'))


@app.route('/view_database')
@login_required  # optional
def view_database():
    conn = sqlite3.connect('movies_app.db')
    conn.row_factory = sqlite3.Row  # so we can access columns by name
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

    cursor.execute("SELECT * FROM feedback")
    feedback = cursor.fetchall()

    conn.close()

    return render_template('view_database.html', users=users, feedback=feedback)


@app.route('/get_feedback')
@login_required
def get_feedback():
    user_id = current_user.id  # Must be logged in
    conn = sqlite3.connect('movies_app.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT book_id, liked, feedback_time
        FROM feedback
        WHERE user_id = ?
        ORDER BY feedback_time DESC
        LIMIT 5
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()

    feedback_data = [
        {
            'book': row[0],
            'liked': bool(row[1]),
            'timestamp': row[2]
        } for row in rows
    ]

    return jsonify(feedback_data)


@app.route('/like', methods=['POST'])
@login_required
def like_book():
    book_title = request.form.get('title')
    username = current_user.username

    if not book_title or not username:
        return jsonify({'status': 'error', 'message': 'Missing data'})

    index = df[df['title'] == book_title].index
    if not index.empty:
        idx = index[0]
        feedback = df.at[idx, 'User_Feedback']
        try:
            feedback_dict = ast.literal_eval(feedback) if feedback else {}
        except:
            feedback_dict = {}

        feedback_dict[username] = 1  # 1 = liked
        df.at[idx, 'User_Feedback'] = str(feedback_dict)
        df.at[idx, 'username'] = username
        df.to_csv(r'D:\phishingurl\goodreads_data.csv', index=False)

        return jsonify({'status': 'success', 'message': 'title liked'})
    return jsonify({'status': 'error', 'message': 'title not found'})


@app.route('/dislike', methods=['POST'])
@login_required
def dislike_book():
    book_title = request.form.get('title')
    username = current_user.username

    if not book_title or not username:
        return jsonify({'status': 'error', 'message': 'Missing data'})

    index = df[df['title'] == book_title].index
    if not index.empty:
        idx = index[0]
        feedback = df.at[idx, 'User_Feedback']
        try:
            feedback_dict = ast.literal_eval(feedback) if feedback else {}
        except:
            feedback_dict = {}

        feedback_dict[username] = 0  # 0 = disliked
        df.at[idx, 'User_Feedback'] = str(feedback_dict)
        df.at[idx, 'username'] = username
        df.to_csv(r'D:\phishingurl\goodreads_data.csv', index=False)

        return jsonify({'status': 'success', 'message': 'title disliked'})
    return jsonify({'status': 'error', 'message': 'title not found'})


@app.route('/delete_my_feedback', methods=['POST'])
def delete_my_feedback():
    if not current_user.is_authenticated:
        return "User not logged in", 401

    conn = sqlite3.connect('movies_app.db')
    cursor = conn.cursor()

    username = current_user.username
    user_id = database.add_user(username)

    cursor.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    # Redirect back to dashboard after deletion
    return redirect(url_for('dashboard'))


@app.route('/feedback', methods=['POST'])
def receive_feedback():
    if not current_user.is_authenticated:
        return jsonify({'error': 'User not logged in'}), 401

    data = request.get_json()
    book_title = data.get('book')
    feedback = data.get('feedback')
    similar_books_titles = data.get('similar_books_titles', [])

    if book_title is None or feedback not in [0, 1]:
        return jsonify({'error': 'Invalid data'}), 400

    global df, final_liked, final_disliked

    username = current_user.username
    user_id = database.add_user(username)  # or get_user_id(username) if exists

    # Get book details from df
    book_row = df[df['normalized_title'] == normalize_title(book_title)]
    original_book_author = book_row['director'].values[0] if not book_row.empty else ""

    genres = []
    if not book_row.empty:
        genres_raw = book_row.iloc[0]['genres']
        try:
            genres = ast.literal_eval(genres_raw) if isinstance(
                genres_raw, str) and genres_raw.startswith('[') else [genres_raw]
            genres = [g.strip().lower() for g in genres]
        except:
            genres = []

    # Store feedback in the database
    database.add_feedback(user_id, book_title, feedback, genres)

    # Mark the feedback in the DataFrame
    df.loc[df['title'] == book_title, 'User_Feedback'] = feedback
    df.loc[df['title'] == book_title, 'username'] = username

    # Get user's liked/disliked genres from DB
    liked_genres, disliked_genres = database.get_liked_books_and_genres(
        user_id)

    # Removed session fallback  only use DB data
    # If new user with no data, genres will be empty

    # Count genre feedback
    liked_counts = dict(Counter(liked_genres))
    disliked_counts = dict(Counter(disliked_genres))

    final_liked = set()
    final_disliked = set()
    all_genres = set(liked_counts.keys()).union(disliked_counts.keys())

    for genre in all_genres:
        if liked_counts.get(genre, 0) > disliked_counts.get(genre, 0):
            final_liked.add(genre)
        elif disliked_counts.get(genre, 0) > liked_counts.get(genre, 0):
            final_disliked.add(genre)

    final_liked = set(list(final_liked)[:5])  # Limit to top 5 liked genres

    # Recommend one book per liked genre
    genre_based_books = []
    seen_books = set()
    similar_books_titles_set = set(similar_books_titles)

    for genre in final_liked:
        matched_books = df[df['genres'].apply(
            lambda g: genre.lower() in [
                x.lower() for x in (
                    ast.literal_eval(g) if isinstance(
                        g, str) and g.startswith('[') else [g]
                )
            ]
        )]

        filtered_books = matched_books[
            (matched_books['director'] != original_book_author) &
            (~matched_books['title'].isin(seen_books)) &
            (~matched_books['title'].isin(similar_books_titles_set))
        ]

        sorted_books = filtered_books.sort_values(by='score', ascending=False)

        if not sorted_books.empty:
            top_book = sorted_books.iloc[0]
            genre_based_books.append(top_book)
            seen_books.add(top_book['title'])

    # final list
    global prev_similar_list_global
    prev_similar_list_global = [{
        'title': row['title'],
        'desc': row['keywords'],
        'rating': row['vote_average'],
        'img': book_url
    } for _, row in pd.DataFrame(genre_based_books).iterrows()]

    return jsonify({'message': 'Feedback processed successfully'})


if __name__ == '__main__':
    database.add_password_hash_column()
    add_initial_user('user1', 'User@1')
    app.run(debug=True)

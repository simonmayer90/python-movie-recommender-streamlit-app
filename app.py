# imports
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# %% load data
movie_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/movies.csv')
rating_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/ratings.csv')
links_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/links.csv')
tags_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/tags.csv')

# %% format dataframes
# MOVIE DF:
movie_df = (
    movie_df
        .assign(year=lambda df_ : df_['title'].replace(r'(.*)\((\d{4})\)', r'\2', regex= True))
        # replace with 0 if there is no year
        .assign(year=lambda df_ : np.where(df_['year'].str.len() <=5 , df_['year'], 0)))
# convert the year column to int
movie_df['year'] = movie_df['year'].astype(int)

# create a genre list
genre_list = []
for i in movie_df['genres']:
    if "|" in i:
        genre_list.extend(i.rsplit("|"))
    else:
        genre_list.append(i)
genre_list = list(set(genre_list))

i = genre_list.index("(no genres listed)")
del genre_list[i]
genre_list.sort()
genre_list.insert(0, 'Any')

year_list = list(set(list(movie_df['year'])))[1:]


# create a list of movies
movie_list = list(set(list(movie_df['title'])))

# %% RATING DF
# convert timestamp to datetime format
rating_df['datetime'] = rating_df['timestamp'].apply(datetime.fromtimestamp)
# drop the timestamp column
rating_df.drop(columns=['timestamp'], inplace=True)
# %% DEFINE FUNCTIONS


# to make the the dataframe look nicer
def make_pretty(styler):
    styler.set_caption("Top movie recommendations for you")
    # styler.background_gradient(cmap="YlGnBu")
    return styler

# population based
def popular_n_movies(n, genre):
    popular_n = (
    rating_df
            .groupby(by='movieId')
            .agg(rating_mean=('rating', 'mean'),
                 rating_count=('movieId', 'count'),
                 datetime=('datetime','mean'))
            .sort_values(['rating_mean','rating_count','datetime'], ascending= False)
            .loc[lambda df_ :df_['rating_count'] >= (df_['rating_count'].mean() + df_['rating_count'].median())/2]
            .reset_index()
    )['movieId'].to_list()
    result = movie_df.loc[lambda df_ : df_['movieId'].isin(popular_n)]
    if genre != 'Any':
            result = result.loc[lambda df_ : df_['genres'].str.contains(genre)]
    df_rec = result.head(n).reset_index(drop=True)
    df_rec = df_rec[['title', 'genres', 'year']].reset_index(drop=True)
    new_index = ['movie-{}'.format(i+1) for i in range(n)]
    df_rec.index = new_index
    pretty_rec = df_rec.style.pipe(make_pretty)
    return pretty_rec

# movie/item based
def item_n_movies(movie_name, n):
    min_rate_count = 10
    movieId = list(movie_df[movie_df['title'] == movie_name].movieId.head(1))[0]

    movies_crosstab = pd.pivot_table(data=rating_df, values='rating',
                                     index='userId',
                                     columns='movieId')

    movie_ratings = movies_crosstab[movieId]
    movie_ratings = movie_ratings[movie_ratings>=0] # exclude NaNs

    # evaluating similarity
    similar_to_movie = movies_crosstab.corrwith(movie_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['PearsonR'])
    corr_movie.dropna(inplace=True)

    rating = pd.DataFrame(rating_df.groupby('movieId')['rating'].mean())
    rating['rating_count'] = rating_df.groupby('movieId')['rating'].count()
    movie_corr_summary = corr_movie.join(rating['rating_count'])
    movie_corr_summary.drop(movieId, inplace=True) # drop forrest gump itself

    top_n = movie_corr_summary[movie_corr_summary['rating_count'] >= min_rate_count].sort_values('PearsonR', ascending=False).head(n)
    top_n = top_n.merge(movie_df, left_index=True, right_on="movieId")
    top_n = top_n[['title', 'genres']].reset_index(drop=True)
    new_index = ['movie-{}'.format(i+1) for i in range(n)]
    top_n.index = new_index
    pretty_rec = top_n.style.pipe(make_pretty)
    return pretty_rec

# user based
def user_n_movies(user_id, n):
    users_items = pd.pivot_table(data=rating_df,
                                      values='rating',
                                      index='userId',
                                      columns='movieId')

    users_items.fillna(0, inplace=True)

    user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                      columns=users_items.index,
                                      index=users_items.index)

    weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))
    not_seen_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_seen_movies.T.dot(weights), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movie_df, left_index=True, right_on="movieId")
    top_recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)
    top_recommendations = top_recommendations[['title', 'genres']].reset_index(drop=True)
    new_index = ['movie-{}'.format(i+1) for i in range(n)]
    top_recommendations.index = new_index
    pretty_rec = top_recommendations.style.pipe(make_pretty)
    return pretty_rec



# i will write another version of this function can manage time period of movies too
x = popular_n_movies(5, 'Any')

# %% STREAMLIT
# Set configuration
st.set_page_config(page_title="WBSFLIX",
                   page_icon="🎬",
                   initial_sidebar_state="expanded",
                   # layout="wide"
                   )

# set colors: These has to be set on the setting menu online
    # primary color: #FF4B4B, background color:#0E1117
    # text color: #FAFAFA, secindary background color: #E50914

# Set the logo of app
st.sidebar.image("D:/Dev/wbs-ds-projekte/git-versions/WBSFlix-Recommender-System/app/wbs_logo.png",
                 width=300, clamp=True)
welcome_img = Image.open('D:/Dev/wbs-ds-projekte/git-versions/WBSFlix-Recommender-System/app/welcome_page_img01.png')
st.image(welcome_img)
st.markdown("""
# 🎬 Welcome to the next generation movie recommendation app
""")

# %% APP WORKFLOW
st.markdown("""
### How may we help you?
"""
)
# Popularity based recommender system
genre_default, n_default = None, None
pop_based_rec = st.checkbox("Show me the all time favourites",
                            False,
                            help="Movies that are liked by many people")


if pop_based_rec:
    st.markdown("### Select Genre and Nr of movie recommendations")
    genre_default, n_default = None, 5
    with st.form(key="pop_form"):
        # genre_default, year_default = ['Any'], ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        key="n",
                        help="How many movie recommendations would you like to receive?",
                        )

        submit_button_pop = st.form_submit_button(label="Submit")


    if submit_button_pop:
        popular_movie_recs = popular_n_movies(nr_rec, genre[0])
        st.table(popular_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

item_based_rec = st.checkbox("Show me a movie like this",
                             False,
                             help="Input some movies and we will show you similar ones")
from random import choice
short_movie_list = ['Prestige, The (2006)', 'Toy Story (1995)',
                    'No Country for Old Men (2007)']
if item_based_rec:
    st.markdown("### Tell us a movie you like:")
    with st.form(key="movie_form"):
        movie_name = st.multiselect(label="Movie name",
                                    # options=movie_list,
                                    options=pd.Series(movie_list),
                                    help="Select a movie you like",
                                    key='item_select',
                                    default=choice(short_movie_list))

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec_movie",
                           help="How many movie recommendations would you like to receive?",
                           )

        submit_button_movie = st.form_submit_button(label="Submit")

    if submit_button_movie:
        st.write('Because you like {}:'.format(movie_name[0]))

        item_movie_recs = item_n_movies(movie_name[0], nr_rec)
        st.table(item_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

user_based_rec = st.checkbox("I want to get personalized recommendations",
                             False,
                             help="Login to get personalized recommendations")

if user_based_rec:
    st.markdown("### Please login to get customized recommendations just for you")
    genre_default, n_default = None, 5
    with st.form(key="user_form"):

        user_id = st.number_input("Please enter your user id", step=1,
                                  min_value=1)
        # genre_default, year_default = ['Any'], ['Any']
        # genre = st.multiselect(
        #         "Genre",
        #         options=genre_list,
        #         help="Select the genre of the movie you would like to watch",
        #         default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec",
                           help="How many movie recommendations would you like to receive?",
                           )

        submit_button_user = st.form_submit_button(label="Submit")


    if submit_button_user:
        user_movie_recs = user_n_movies(user_id, nr_rec)
        st.table(user_movie_recs)

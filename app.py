
# imports
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from datetime import datetime
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
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
def get_popular_recommendations(n, genres):
    if genres == "Any":
        genres = ""
    recommendations = (
        rating_df
            .groupby('movieId')
            .agg(avg_rating = ('rating', 'mean'), num_ratings = ('rating', 'count'))
            .merge(movie_df, on='movieId')
            .assign(combined_rating = lambda x: x['avg_rating'] * x['num_ratings']**0.5)
            [lambda df: df["genres"].str.contains(genres, regex=True)]
#             .loc[lambda df : ((df['year'] >= time_range[0]) & ( df['year'] <= time_range[1]))]
            .sort_values('combined_rating', ascending=False)
            .head(n)
            [['title', 'avg_rating', 'genres']]
            .rename(columns= {'title': 'Movie Title', 'avg_rating': 'Average Rating', 'genres': 'Genres'})
    )
    pretty_recommendations = recommendations.style.pipe(make_pretty)
    return pretty_recommendations


def popular_n_movies(n, genre):
    popular_n = (
    rating_df
            .groupby(by='movieId')
            .agg(rating_mean=('rating', 'mean'),
                 rating_count=('movieId', 'count'),
                 datetime=('datetime','mean'))
            .assign(combined_rating = lambda x: x['avg_rating'] * x['num_ratings']**0.5)
            .sort_values(['combined_rating','datetime'], ascending= False)
#             .loc[lambda df_ :df_['rating_count'] >= (df_['rating_count'].mean() + df_['rating_count'].median())/2]
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

# user based with year as input
def top_n_user_based(user_id , n , genres, time_period):
    if user_id not in rating_df["userId"]:
        return pd.DataFrame(columns= ['movieId', 'title', 'genres', 'year'])

    users_items = pd.pivot_table(data=rating_df,
                                 values='rating',
                                 index='userId',
                                 columns='movieId')
    users_items.fillna(0, inplace=True)
    user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index,
                                 index=users_items.index)
    weights = (
    user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
          )

    new_userids = weights.sort_values(ascending=False).head(100).index.tolist()
    new_userids.append(user_id)
    new_ratings = rating_df.loc[lambda df_: df_['userId'].isin(new_userids)]
    new_users_items = pd.pivot_table(data=new_ratings,
                                 values='rating',
                                 index='userId',
                                 columns='movieId')

    new_users_items.fillna(0, inplace=True)
    new_user_similarities = pd.DataFrame(cosine_similarity(new_users_items),
                                         columns=new_users_items.index,
                                         index=new_users_items.index)
    new_weights = (
    new_user_similarities.query("userId!=@user_id")[user_id] / sum(new_user_similarities.query("userId!=@user_id")[user_id])
          )
    not_watched_movies = new_users_items.loc[new_users_items.index!=user_id, new_users_items.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_watched_movies.T.dot(new_weights), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movie_df, left_index=True, right_on="movieId").sort_values("predicted_rating", ascending=False)
    recommendations = recommendations.loc[lambda df_ : ((df_['year'] >= time_period[0]) & ( df_['year'] <= time_period[1]))]
    if len(genres)>0:
        result = pd.DataFrame(columns=['predicted_rating', 'movieId', 'title', 'genres', 'year'])
        for genre in genres:
            result = pd.concat([result, recommendations.loc[lambda df_ : df_['genres'].str.contains(genre)]])

        result.drop_duplicates(inplace=True)
        result = result.sort_values("predicted_rating", ascending=False)
        result.reset_index(inplace=True, drop= True)
        return result.drop(columns=['predicted_rating']).head(n)

    return recommendations.reset_index(drop=True).drop(columns=['predicted_rating']).head(n)
# %% STREAMLIT
# Set configuration
st.set_page_config(page_title="WBSFLIX",
                   page_icon="🎬",
                   initial_sidebar_state="expanded",
                   layout="wide"
                   )

# set colors: These has to be set on the setting menu online
    # primary color: #FF4B4B, background color:#0E1117
    # text color: #FAFAFA, secondary background color: #E50914

# Set the logo of app
st.sidebar.image("wbs_logo.png",
                 width=300, clamp=True)
welcome_img = Image.open('welcome_page_img01.png')
st.image(welcome_img)
st.sidebar.markdown("""
# 🎬 Welcome to the next generation movie recommendation app
""")

# %% APP WORKFLOW
st.sidebar.markdown("""
### How may we help you?
"""
)
# Popularity based recommender system
genre_default = None
pop_based_rec = st.sidebar.checkbox("Show me the all time favourites",
                            False,
                            help="Movies that are liked by many people")


if pop_based_rec:
    st.markdown("### Select the Genre and the Number of recommendations")
    genre_default, n_default = None, 5
    with st.form(key="pop_form"):
        genre_default = ['Any']
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
#         popular_movie_recs = get_popular_recommendations(nr_rec, genre[0])
        popular_movie_recs = popular_n_movies(nr_rec, genre[0])
        st.table(popular_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

item_based_rec = st.sidebar.checkbox("Show me a movie like this",
                             False,
                             help="Input some movies and we will show you similar ones")

if item_based_rec:
    st.markdown("### Tell us a movie you like:")
    with st.form(key="movie_form"):
        movie_name = st.multiselect(label="Movie name",
                                    # options=movie_list,
                                    options=pd.Series(movie_list),
                                    help="Select a movie you like",
                                    key='item_select',
                                    # default=choice(short_movie_list)
                                    )

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

user_based_rec = st.sidebar.checkbox("I want to get personalized recommendations",
                             False,
                             help="Login to get personalized recommendations")

if user_based_rec:
    st.markdown("### Please login to get customized recommendations just for you")
    genre_default, n_default = None, 5
    with st.form(key="user_form"):

        user_id = st.number_input("Please enter your user id", step=1,
                                  min_value=1)
        genre_default = ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                #default=genre_default
                )

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec",
                           help="How many movie recommendations would you like to receive?",
                           )

        time_period = st.slider('years:', min_value=1900,
                                max_value=2018,
                                value=(2010,2018),
                                step=1)

        submit_button_user = st.form_submit_button(label="Submit")

    if submit_button_user:
        # user_movie_recs = user_n_movies(user_id, nr_rec)
        user_movie_recs = top_n_user_based(user_id, nr_rec, genre, time_period)

        # st.write(time_period)
        st.table(user_movie_recs[['title', 'genres']].style.pipe(make_pretty))

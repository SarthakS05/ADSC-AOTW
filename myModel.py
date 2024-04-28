import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials 
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='0c7c294438794162b901ce0554a38703', client_secret='cecd70a59c1344f7a7d8d7cbec4fe6bb'))

#uncomment above and replace CLIENT_ID and CLIENT_SECRET with the id and secret you get from creating your app with spotify (going through the "Getting Started" instructions in the Spotify Web API documentation)

from sklearn.model_selection import train_test_split
import math

discussions_df = pd.read_csv('data/Discussions.csv', header=0, encoding='utf-8')
ratings_df = pd.read_csv('data/Ratings.csv', header=0, encoding='utf-8')
# TODO: integrate Spotify with your filtering code
discussions_df['energy'] = 0
for ind in range(len(discussions_df)):
    try:
        results = spotify.album_tracks(discussions_df['SpotifyID'][ind]) # getting tracks from current album via SpotiPy

        # filtering track ids into list
        ids = []
        for track in results['items']:
            ids.append(track['id'])
        numTracks = len(ids)
        features = spotify.audio_features(ids)
        for feature in features:
            # calculating and storing average in df
        
            discussions_df['energy'][ind] += (feature['speechiness'] + feature['Rating'])
    except:
        pass
    
lengthy = discussions_df[['DiscussionID','AlbumName', 'energy', 'AvgRating']]

ratings_df['energy'] = lengthy['energy']

X = ratings_df.drop(columns=['energy'])
y = ratings_df['energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train['energy'] = y_train

merged_df = pd.merge(lengthy, X_train, on='DiscussionID')

display(merged_df[["DiscussionID", "AlbumName", "energy_x", "Rating"]])
y_pred = []
for idx, row in X_test.iterrows():
    # Get the decade of the current album
    album_decade = discussions_df.loc[discussions_df['DiscussionID'] == row['DiscussionID'], 'energy'].iloc[0]
    
    # Filter merged_df to only include discussions attended by the user and in the same decade
    smaller_df = merged_df[(merged_df['MemberID'] == row['MemberID'])]
    
    # Calculate the average rating for discussions in the same decade
    try:
        avg_rating = smaller_df['energy_y'].mean() if not smaller_df.empty else X_train['energy_y'].mean()
    except:
        pass
    y_pred.append(float(np.round(avg_rating, 0)))

# Calculate RMSE
y_pred = np.array(y_pred)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print("RMSE:", rmse)


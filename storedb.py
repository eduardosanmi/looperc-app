import streamlit as st
import pandas as pd
from qdrant_client import models, QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

audio_collection_name = st.secrets["DB_C1"]
meta_collection_name = st.secrets["DB_C2"]
qdrant = QdrantClient(
    st.secrets["DB_URL"],
    api_key=st.secrets["DB_API_KEY"],
)

def get_samples_by_condition_qdrant(df = None, genres = None,
 selected_bpm = None, selected_genre = None, bpm_range = None, 
 shuffle = True, album = None, token=None, limit=250):
    
    conditions = []
    if selected_genre:
        conditions.append(FieldCondition(key="genre", match=MatchAny(any=selected_genre)))
    if selected_bpm:
        conditions.append(FieldCondition(key="bpm", match=MatchValue(value=selected_bpm)))  
    if token:
        conditions.append(FieldCondition(key="token", match=MatchValue(value=token)))
    if album:
        conditions.append(FieldCondition(key="album", match=MatchValue(value=album)))
    if not conditions:
        print("Error: Please provide at least one filter condition.")
        return None    
    
    try:
        results_df, results = query_by_payload( 
            genre = selected_genre, bpm = selected_bpm, token = token, album = album, limit = limit)
        df_results_audio = query_by_id( results_df )
        df_llm = results_df.join( df_results_audio )
    except:
        print("Error: Something went wrong.")
        return None        
    
    return df_llm

def query_by_id(results_df):
    result_df_audio = []
    for i in list(results_df.index):
        result_df_audio.append ( qdrant.scroll(limit = 5, collection_name=audio_collection_name, # with_payload = [],
                      scroll_filter=models.Filter(must=[models.HasIdCondition(has_id = [i]),],),)[0][0] )

    results_data_audio = [result.payload for result in result_df_audio]
    results_index_audio = [result.id for result in result_df_audio]    
    df_results_audio = pd.DataFrame(results_data_audio)
    df_results_audio.index = results_index_audio
    
    return df_results_audio

def query_by_payload(genre=None, bpm=None, token=None, album=None, limit=5):
    conditions = []
    if genre:
        conditions.append(FieldCondition(key="genre", match=MatchAny(any=genre)))
    if bpm:
        conditions.append(FieldCondition(key="bpm", match=MatchValue(value=bpm)))
    if token:
        conditions.append(FieldCondition(key="token", match=MatchValue(value=token)))
    if album:
        conditions.append(FieldCondition(key="album", match=MatchValue(value=album)))
    if not conditions:
        print("Error: Please provide at least one filter condition.")
        return None
    
    query_filter = Filter(must=conditions)    
    search_results, next_offset = qdrant.scroll(
        collection_name=meta_collection_name,
        scroll_filter=query_filter,
        limit=limit
    )
    
    if not search_results:
        print("No matching results found!")
        return None

    results_data = [result.payload for result in search_results]
    results_index = [result.id for result in search_results]    
    df_results = pd.DataFrame(results_data)
    df_results.index = results_index
    
    return df_results, search_results

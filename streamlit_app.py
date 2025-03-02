import streamlit as st
import numpy as np
from storedb import query_by_payload
from sampler import gen_audio, genres
import time

avail_albums = query_by_payload(genre = genres, limit = 1000)

st.title("l:blue[OO]perc")
st.subheader(":grey[Jam :blue[fresh] new tracks...]", divider="gray")

sel_selected_album = st.selectbox("selection",
                                  list(avail_albums[0].album.unique()),
                                  index=None,
                                  label_visibility = 'collapsed')
if sel_selected_album is None:
    st.warning("Please select an option to continue.")
    st.stop()

with st.spinner("Generating track, this can take a while... ...", show_time=True): 
    data_wav, sections, durations = gen_audio(
        album = sel_selected_album, n_steps = 6)

st.audio(np.array(data_wav).T, sample_rate=44100, loop = False, autoplay = True)
with st.empty(): 
    time.sleep(.2)
    for i in range(len(sections)):
        columns = st.columns((1, 1, 1)) 
        with columns[1]:
            st.subheader(f''':gray[Playing :green[{sections[i]}] for {round(durations[i],1)} secs ]''')
            if i < len(sections)-1:
                st.write(f''':gray[Next :green[{sections[i+1]}] for {round(durations[i+1],1)} secs ]''')
            time.sleep(durations[i])    
    

        
    







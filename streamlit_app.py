import streamlit as st
import numpy as np
from storedb import query_by_payload
from sampler import gen_audio, genres
import time, random

st.title("l:blue[OO]perc")
st.subheader(":grey[Jam :violet[fresh] new tracks...]", divider="gray")

sel_selected_genre = st.selectbox("selection",
                                  genres,
                                  index=None,
                                  label_visibility = 'collapsed')
if sel_selected_genre is None:
    st.warning("Please select an option to continue.")
    st.stop()

avail_albums = query_by_payload(genre = [sel_selected_genre], limit = 100)

if avail_albums:
    jam_button = st.button(f'''Jam! ({len(avail_albums[0].album.unique())})''', type="primary")
    if jam_button:
        sel_selected_album = random.choice(list(avail_albums[0].album.unique())) 
        with st.spinner("Generating track, this can take a while... ...", show_time=True): 
            data_wav, sections, durations = gen_audio(
                album = sel_selected_album, n_steps = 6)
            st.audio(np.array(data_wav).T, sample_rate=44100, loop = False, autoplay = True)
        with st.empty():            
            for i in range(len(sections)):
                columns = st.columns(1) 
                with columns[0]:
                    st.subheader(f''':gray[Playing :green[{sections[i].upper()}] for :green[{round(durations[i],1)} secs]]''')
                    if i < len(sections)-1:
                        time.sleep(0.101)
                        st.write(f''':gray[Next :violet[{sections[i+1].upper()}] for :violet[{round(durations[i+1],1)} secs ]]''')
                    time.sleep(durations[i])     
else:
    jam_button = st.button("Jam! (0)", type="secondary", disabled=True)
    


        
    







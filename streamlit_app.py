import streamlit as st
import numpy as np
from storedb import query_by_payload
from sampler import gen_audio

st.title("l:blue[OO]perc")
st.subheader(":grey[Generate :blue[fresh] new tracks...]", divider="gray")

avail_albums = query_by_payload(genre = ['funk','rock','reggae','blues','jazz'], 
                                limit = 1000)
sel_selected_album = st.selectbox(
    "album",
    list(avail_albums[0].album.unique()),
    index=None,
    label_visibility = 'collapsed')

if sel_selected_album is None:
    st.warning("Please select an option to continue.")
    st.stop()

with st.spinner("Generating loop, this can take a while... ...", show_time=True): 
    data_wav = gen_audio(album = sel_selected_album, n_steps = 3)
    st.audio(np.array(data_wav).T, sample_rate=44000)







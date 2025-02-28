import streamlit as st

from storedb import query_by_payload
from sampler import gen_audio, generate_audio_html 

st.title("l:blue[OO]perc")
st.subheader(":grey[Each time will be a :blue[fresh] new track...]", divider="gray")

results_q = query_by_payload(genre = ['funk','rock','reggae','blues'], limit = 10000)

sel_selected_album = st.selectbox(
    "album",
    list(results_q[0].album.unique()),
    index=None,
    label_visibility = 'collapsed')

if sel_selected_album is None:
    st.warning("Please select an option to continue.")
    st.stop()
# st.write("Your Selection:", sel_selected_album)

with st.spinner("Generating loop, this can take a while... ...", show_time=True): 
    data_wav = gen_audio(album = sel_selected_album, n_steps = 3)
    audio_placeholder = st.empty()
    audio_html = generate_audio_html(data_wav)
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)






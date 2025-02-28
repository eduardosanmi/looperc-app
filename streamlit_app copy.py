import streamlit as st
import io
import base64
import soundfile as sf
from storedb import query_by_payload
from sampler import gen_audio 

st.title("lOOperc")

results_q = query_by_payload(genre = ['funk','rock'], limit = 10000)
sel_selected_album = st.selectbox(
    "album",
    list(results_q[0].album.unique()),
    index=None,
    label_visibility = 'collapsed')
if sel_selected_album is None:
    st.warning("Please select an option to continue.")
    st.stop()
st.write("Selected Album:", sel_selected_album)

with st.spinner("Generating loop.... this can take a while... ...", show_time=True): 
    data_wav = gen_audio(album = sel_selected_album, n_steps = 2)
    '''    
    audio_template = st.secrets["audio"]["HTML_TEMPLATE"]
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, data_wav, 44100, format="WAV")  
    audio_buffer.seek(0)  
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    audio_html = audio_template.format(audio_base64=audio_base64)
    st.markdown(audio_html, unsafe_allow_html=True)
    '''

def generate_audio_html(audio_array):
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_array, 44100, format="WAV")
    audio_buffer.seek(0)  
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    audio_html = f"""
    <audio controls controlsList="nodownload">
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

audio_placeholder = st.empty()
audio_html = generate_audio_html(data_wav)
audio_placeholder.markdown(audio_html, unsafe_allow_html=True)




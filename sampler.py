import streamlit as st

import pandas as pd
import numpy as np
from itertools import product
import random
import copy
import io
import base64
import soundfile as sf
import gc
from storedb import get_samples_by_condition_qdrant

standard_song_patterns = st.secrets["STANDAR_SONG_PATTERNS"]
genres = ["funk","garage", "rock", "country", "blues", "jazz", "latin", "reggae", "brushes", "upbeat", "boggie"]

def generate_audio_html(audio_array):
    audio_template = st.secrets["audio"]["HTML_TEMPLATE"]
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_array, 44100, format="WAV")
    audio_buffer.seek(0)  
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    del audio_array
    gc.collect() 
    audio_html = audio_template.format(audio_base64=audio_base64)
    return audio_html

def gen_audio(bpm = None, genre = None, album = None, n_steps = 5):
    sel_selected_album = album
    sel_selected_genre = genre
    sel_selected_bpm = bpm
    audio_lib_df = None
    genres = None
    sel_bpm_range = 0
    sel_shuffle = True
    sel_topK = 15
    round_val = 1
    use_llm = False
    model_simple = False
    sel_model = None

    loop_struct_dict_array, df_llm = gen_loop_content(
        audio_lib_df, genres, sel_selected_bpm, sel_selected_genre, 
        sel_bpm_range, sel_shuffle, sel_selected_album,
        sel_topK, round_val, use_llm = use_llm, 
        model_simple = model_simple, sel_model = sel_model)
    
    steps_dict = []
    data = None
    prev_data = None
    two_prev = True

    loop_struct_dict_array_ini = get_intro_patterns(loop_struct_dict_array)
    if len(loop_struct_dict_array_ini) == 0:
        loop_struct_dict_array_ini = loop_struct_dict_array
    
    data_wav_tmp = []
    data, sampled_element, found_pattern = sample_from_dict(
        loop_struct_dict_array_ini, df_llm, round_val)
    step_dict = {'found_pattern': found_pattern,
                'tokens': sampled_element.token.to_list(),
                'duration': sampled_element.duration.to_list(),
                'files': sampled_element.file.to_list(),
                'album': list(sampled_element.album.unique())}
    steps_dict.append(step_dict)
    data_wav_tmp.append(data)
        
    for step_ix in range(n_steps):
        data, sampled_element, prev_data, step_dict = data_stream_pull(
            sampled_element, loop_struct_dict_array, df_llm, round_val, 
            data = data, prev_data = prev_data, two_prev = two_prev, steps_dict = steps_dict) 
        data_wav_tmp.append(data)
    
    del df_llm
    gc.collect()
    
    data_wav = np.concatenate(data_wav_tmp)
    sections = [x for i in range(len(steps_dict)) for x in steps_dict[i]['tokens']]
    durations = [x for i in range(len(steps_dict)) for x in steps_dict[i]['duration']]

    return data_wav, sections, durations 


# calc time for a bar given BPM
def bar_t_calc(BPM, pulse_Compas=4, TR=60):
    bar_t = (TR * pulse_Compas) / BPM
    return bar_t

# calc time for a n-bar given bar time and nº of bars
def set_song_length(bar_t, N_bars):
    return bar_t * N_bars

# subset sum naive algo
def subset_sum(numbers, target, partial=[], partial_sum=0, margin = .1):
    if (target - margin) <= partial_sum <= (target + margin):
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)

# generate combination of time patterns that sum a given total time        
def generate_sample_time_patterns(df_llm, total_length, 
                                  topK = 3, pattern_precision = 2):
    sorted_available_lengths = df_llm.groupby(
        df_llm['duration'].round(pattern_precision)).size().sort_values(ascending = False)

    lengths = sorted_available_lengths.iloc[0:topK].index.values
    repeats = ((total_length/lengths).round()).astype(int)
    available_lens = [x for sublist in [ 
        [lengths[x]] * repeats[x] for x in range(len(lengths)) ] for x in sublist]
    
    ## subset sum
    time_patterns = []
    subset = subset_sum(available_lens, round(total_length, pattern_precision))
    [time_patterns.append(x) for x in list(subset) if x not in time_patterns]
    time_patterns_copy = copy.deepcopy(time_patterns[:])
    [i.reverse() for i in time_patterns_copy]
    [time_patterns.append(i) for i in time_patterns_copy if i not in time_patterns]

    return time_patterns

# traslate time patterns to token patterns
def posible_patterns(sample_time_patterns, df_llm, round_val):
    possible_bar_patterns = []
    for sel_sample_time_patterns in range(len(sample_time_patterns)):
        current_pattern = sample_time_patterns[sel_sample_time_patterns]
        time_matching_loops = [ list(df_llm[ round(df_llm.duration, round_val) == current_pattern[i]].token.unique()) 
                               for i in range(len(current_pattern))]
        possible_bar_patterns.append(time_matching_loops)
    
    return possible_bar_patterns

# Get all token combination for a sample time 
def llm_res_files_by_cartesian(possible_bar_patterns_to_prompt):
    sequences = []
    for i in range(len(possible_bar_patterns_to_prompt)):
        sequences.append( { str(i) : list(product(*possible_bar_patterns_to_prompt[i])) } )
    
    llm_res_files = {'sequences': sequences }
    
    return llm_res_files

def gen_struct_dict(sample_time_patterns, llm_res_files, standard_song_patterns = standard_song_patterns):
    sequences = llm_res_files['sequences']
    
    assert len(sample_time_patterns) == len( sequences )
    for i in range(len(sample_time_patterns)):
        for v in range(len( sequences[i][str(i)] )):
            assert len(sample_time_patterns[i]) == len(sequences[i][str(i)][v]) 
            
    loop_strcut_dict_array = []
    for i in range(len(sample_time_patterns)):
        loop_strcut_dict_array_sub_ver = []
        for v in range(len( sequences[i][str(i)] )):
            loop_strcut_dict_array_sub = []
            song_sub_pattern = " → ".join(sequences[i][str(i)][v])
            if max([song_sub_pattern in st for st in standard_song_patterns]):             
                for n in range(len( sequences[i][str(i)][v] )):
                    loop_strcut_dict_array_sub.append({'duration': sample_time_patterns[i][n],
                                                       'token': sequences[i][str(i)][v][n]})
                loop_strcut_dict_array_sub_ver.append(loop_strcut_dict_array_sub)
        if len(loop_strcut_dict_array_sub_ver):
            loop_strcut_dict_array.append(loop_strcut_dict_array_sub_ver)
        
    return loop_strcut_dict_array 

def get_intro_patterns(data):
    filtered_data = [[sublist for sublist in inner_list if any(item['token'] == 'intro' for item in sublist)]
                     for inner_list in data if any(sublist for sublist in inner_list if any(item['token'] == 'intro' for item in sublist))]

    return filtered_data

def gen_loop_content(audio_lib_df, genres, sel_selected_bpm, sel_selected_genre, sel_bpm_range, sel_shuffle, sel_selected_album,
                    sel_topK, round_val, use_llm = False, model_simple = True, sel_model = None):
    ### START
    loop_struct_dict_array_list = []

    df_llm = get_samples_by_condition_qdrant(selected_bpm = sel_selected_bpm, 
                                       selected_genre = sel_selected_genre, bpm_range = sel_bpm_range, 
                                       shuffle = sel_shuffle, album = sel_selected_album)    
        
    for bars in [1,2,3,4,5,6,7,8,9,12]:    
        loop_struct_dict_array = None
        song_dur = None
        sample_time_patterns = None
        llm_res_files = None        
        sampled_element_c = None

        try:
            sel_n_bars = bars
            song_dur = round( set_song_length( bar_t_calc( df_llm.bpm.unique()[0]), sel_n_bars), 3)
            sample_time_patterns = generate_sample_time_patterns(df_llm, song_dur, topK = sel_topK, pattern_precision = round_val)

            if len(sample_time_patterns) > 0:
                possible_bar_patterns_to_prompt = posible_patterns(sample_time_patterns, df_llm, round_val)
                llm_res_files = llm_res_files_by_cartesian(possible_bar_patterns_to_prompt)
                if llm_res_files:        
                    loop_struct_dict_array = gen_struct_dict(sample_time_patterns, llm_res_files)
            else:                
                pass

            if loop_struct_dict_array:
                loop_struct_dict_array_list.append(copy.deepcopy(loop_struct_dict_array))
                print(f'''done: {bars} bars''')
            else:                
                pass

        except:
            print(f'''failed: {bars} bars''')

    loop_struct_dict_array = [x for y in loop_struct_dict_array_list for x in y]  

    return loop_struct_dict_array, df_llm

def remove_ending_patterns(data):
    filtered_data = [[sublist for sublist in inner_list if not any(item['token'] == 'ending' for item in sublist)]
                     for inner_list in data if any(sublist for sublist in inner_list if not any(item['token'] == 'ending' for item in sublist))]

    return filtered_data

def find_valid_standar_song_pattern(prev_data, loop_struct_dict_array, standard_song_patterns = standard_song_patterns):
    prev_data = ' → '.join(prev_data)
    filetered_loop_struct_dict_array = []
    for n in range(len(loop_struct_dict_array)):
        filetered_loop_struct_dict_array_subp = []
        for i in range(len(loop_struct_dict_array[n])):
            future_data = prev_data + ' → ' +  ' → '.join([ x['token'] for x in loop_struct_dict_array[n][i] ])
            if max([future_data in st for st in standard_song_patterns]):
                filetered_loop_struct_dict_array_subp.append(loop_struct_dict_array[n][i])
        if len(filetered_loop_struct_dict_array_subp) > 0:
            filetered_loop_struct_dict_array.append(filetered_loop_struct_dict_array_subp)
    if len(filetered_loop_struct_dict_array):
        loop_struct_dict_array = filetered_loop_struct_dict_array
        # print(f'''Found pattern: {prev_data} ''')
    else:
        # print(f'''Broken pattern: {prev_data} ''')
        None
        
    return loop_struct_dict_array

def gen_loop_from_dict(loop_struct_dict_array, sel_time_pattern, sel_pattern_version, df_llm, round_val):
    # assert len(loop_struct_dict_array) == len(sample_time_patterns)
    loop_ver_tmp = loop_struct_dict_array[sel_time_pattern][sel_pattern_version]
    df_llm_steps = []
    for itx_ver in range(len(loop_ver_tmp)):
        df_llm_step = df_llm[(round(df_llm.duration, round_val) == loop_ver_tmp[itx_ver]['duration'] ) & 
                             (df_llm.token == loop_ver_tmp[itx_ver]['token']) ]
        df_llm_steps.append(df_llm_step.groupby(['token']).apply(lambda x: x.sample(1)).reset_index(drop=True))

    sampled_element = pd.concat(df_llm_steps)
    
    return sampled_element

def sample_from_dict(loop_struct_dict_array, df_llm, round_val, prev_data = None, next_prev_data = None, 
                     no_ending = True, standard_song_patterns = standard_song_patterns):
    found_pattern = 0
    loop_struct_dict_array_orig = copy.deepcopy(loop_struct_dict_array)
    if no_ending is True:
        loop_struct_dict_array = remove_ending_patterns(loop_struct_dict_array)
        loop_struct_dict_array_orig = copy.deepcopy(loop_struct_dict_array)
    
    if prev_data:
        # print(f'''2-step pattern: {prev_data} ''')
        loop_struct_dict_array = find_valid_standar_song_pattern(prev_data, loop_struct_dict_array, standard_song_patterns = standard_song_patterns)
        found_pattern = 2
        
    if next_prev_data:        
        if loop_struct_dict_array == loop_struct_dict_array_orig:
            # print(f'''1-step pattern: {next_prev_data} ''')
            loop_struct_dict_array = find_valid_standar_song_pattern(next_prev_data, loop_struct_dict_array_orig, standard_song_patterns = standard_song_patterns)
            found_pattern = 1
    
    sel_time_pattern, sel_pattern_version = random.choice([ 
        (i,n) for i in range(len(loop_struct_dict_array)) for n in range(len(loop_struct_dict_array[i])) ])
    
    sampled_element = gen_loop_from_dict(loop_struct_dict_array, sel_time_pattern, sel_pattern_version, df_llm, round_val)
    
    data = np.concatenate(sampled_element.audio_data.to_list(),-1)
    data = np.column_stack((data))
    
    return data, sampled_element, found_pattern

def data_stream_pull(sampled_element, loop_struct_dict_array, df_llm, round_val, 
                     data = None, prev_data = None, two_prev = True, steps_dict = []):
    if prev_data is not None:                        
        prev_data += sampled_element.token.to_list()                        
    else:
        prev_data = sampled_element.token.to_list()
    next_prev_data = sampled_element.token.to_list()
    
    data, sampled_element, found_pattern = sample_from_dict(loop_struct_dict_array, df_llm, round_val, prev_data, next_prev_data)                    
    # data = np.concatenate([data, data_new])

    if two_prev:                        
        prev_data = next_prev_data
    else:
        prev_data = None                    

    step_dict = {'found_pattern': found_pattern,
                 'tokens': sampled_element.token.to_list(),
                 'duration': sampled_element.duration.to_list(),
                 'files': sampled_element.file.to_list(),
                 'album': list(sampled_element.album.unique())}
    
    steps_dict.append(step_dict)
    
    return data, sampled_element, prev_data, steps_dict
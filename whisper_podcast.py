URL = "https://podcasts.apple.com/tw/podcast/id1532218131?i=" 
meta_url = "https://itunes.apple.com/lookup?id=1532218131&country=TW&media=podcast&entity=podcastEpisode"
Model = 'large-v2' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large', 'large-v2']
import whisper
import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

whisper_model = whisper.load_model(Model)

def get_podcast_id(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        print(
            f"Error: Unable to fetch the podcast page. Status code: {response.status_code}")
        return
    metadata = response.json()
    return metadata

def find_audio_url(html: str) -> str:
    # Find all .mp3 and .m4a URLs in the HTML content
    audio_urls = re.findall(r'https://[^\s^"]+(?:\.mp3|\.m4a)', html)

    # If there's at least one URL, return the first one
    if audio_urls:
        pattern = r'=https?://[^\s^"]+(?:\.mp3|\.m4a)'
        result = re.findall(pattern, audio_urls[-1])
        if result:
          return result[-1][1:]
        else:
          return audio_urls[-1]

    # Otherwise, return None
    return None

def get_file_extension(url: str) -> str:
    # Parse the URL to get the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Extract the file extension using os.path.splitext
    _, file_extension = os.path.splitext(path)

    print("url", url, path, file_extension)
    # Return the file extension
    return file_extension

def download_apple_podcast(url: str, output_folder: str = 'downloads'):
    response = requests.get(url)
    if response.status_code != 200:
        print(
            f"Error: Unable to fetch the podcast page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    audio_url = find_audio_url(response.text)

    if not audio_url:
        print("Error: Unable to find the podcast audio url.")
        return

    episode_title = soup.find('span', {'class': 'product-header__title'})

    if not episode_title:
        print("Error: Unable to find the podcast title.")
        return

    episode_title = episode_title.text.strip().replace('/', '-')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"{episode_title}{get_file_extension(audio_url)}")

    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return episode_title, output_file

metadata = get_podcast_id(meta_url)
for i, ep in enumerate(metadata["results"]):
    if i == 0: continue
    n_ep = 106-i
    print(n_ep)
    print(ep["trackId"])
    episode_title, output_file = download_apple_podcast(URL+str(ep["trackId"]))
    language = "Auto detection"  # @param ['Auto detection', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese', 'Castilian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'Flemish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Moldavian', 'Moldovan', 'Mongolian', 'Myanmar', 'Nepali', 'Norwegian', 'Nynorsk', 'Occitan', 'Panjabi', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto', 'Romanian', 'Russian', 'Sanskrit', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Sinhalese', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba']
    verbose = 'Live transcription' #@param ['Live transcription', 'Progress bar', 'None']
    output_format = 'vtt' #@param ['txt', 'vtt', 'srt', 'tsv', 'json', 'all']
    task = 'transcribe' #@param ['transcribe', 'translate']
    temperature = 0.15 #@param {type:"slider", min:0, max:1, step:0.05}
    temperature_increment_on_fallback = 0.2 #@param {type:"slider", min:0, max:1, step:0.05}
    best_of = 5 #@param {type:"integer"}
    beam_size = 8 #@param {type:"integer"}
    patience = 1.0 #@param {type:"number"}
    length_penalty = -0.05 #@param {type:"slider", min:-0.05, max:1, step:0.05}
    suppress_tokens = "-1" #@param {type:"string"}
    initial_prompt = "" #@param {type:"string"}
    condition_on_previous_text = True #@param {type:"boolean"}
    fp16 = True #@param {type:"boolean"}
    compression_ratio_threshold = 2.4 #@param {type:"number"}
    logprob_threshold = -1.0 #@param {type:"number"}
    no_speech_threshold = 0.6 #@param {type:"slider", min:-0.0, max:1, step:0.05}

    verbose_lut = {
        'Live transcription': True,
        'Progress bar': False,
        'None': None
    }

    import numpy as np
    import warnings
    import shutil
    from pathlib import Path

    args = dict(
        language = (None if language == "Auto detection" else language),
        verbose = verbose_lut[verbose],
        task = task,
        temperature = temperature,
        temperature_increment_on_fallback = temperature_increment_on_fallback,
        best_of = best_of,
        beam_size = beam_size,
        patience=patience,
        length_penalty=(length_penalty if length_penalty>=0.0 else None),
        suppress_tokens=suppress_tokens,
        initial_prompt=(None if not initial_prompt else initial_prompt),
        condition_on_previous_text=condition_on_previous_text,
        fp16=fp16,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold
    )

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if Model.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{Model} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    audio_path_local = Path(output_file).resolve()
    print("audio local path:", audio_path_local)

    transcription = whisper.transcribe(
        whisper_model,
        str(audio_path_local),
        temperature=temperature,
        **args,
    )

    # Save output
    whisper.utils.get_writer(
        output_format=output_format,
        output_dir=audio_path_local.parent
    )(
        transcription,
        f'EP{n_ep}'
    )
    if n_ep == 98: break



import torch
from utils.audio import *


def process_audio(audio_path):
    audios, audio_lens, audio_span_tokens = [], [], []
    audio = load_audio(audio_path)
    L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s->4s
    mel_len = L // 160
    if mel_len < 16:
        mel_len = 16
    audio = pad_or_trim(audio.flatten())
    mel = log_mel_spectrogram(audio)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
    audio_len = [audio_len_after_cnn, audio_token_num]
    audios.append(mel)
    audio_lens.append(audio_len)
    audio_span_tokens.append(audio_token_num)
    input_audio_lengths = torch.IntTensor(audio_lens)
    input_audios = torch.stack(audios, dim=0)
    return {"input_audios": input_audios,
            "input_audio_lengths": input_audio_lengths,
            "audio_span_tokens": audio_span_tokens,
            "audio_path": audio_path}

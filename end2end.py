print("loading dependancies!")

# imports
import torch
import torch.utils.data
import torch.nn.functional as F
import time
import os

if True:# dataloader only imports
    import librosa
    from scipy.io.wavfile import write
    from scipy.signal import butter, sosfilt
    import pyworld as pw
    import numpy as np
    import difflib
    
    import pyloudnorm as pyln
    from src.dataloader.stft  import TacotronSTFT
    from src.dataloader.utils import load_wav_to_torch

if True:# model only imports
    from src.diffsvc   .model import load_model as init_model_diffsvc
    from src.dilatedasr.model import load_model as init_model_dilated_asr
    from src.hifigan_ct.model import load_generator_from_path as load_model_hifigan

print("Done!")

def get_stft(config):
    stft = TacotronSTFT(config.filter_length, config.hop_length, config.win_length,
                                  config.n_mel_channels, config.sampling_rate, config.mel_fmin,
                                  config.mel_fmax, clamp_val=config.stft_clamp_val)
    return stft

def load_diffsvc_from_path(checkpoint_path, device='cuda'):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model = init_model_diffsvc(checkpoint_dict['h'], device=device)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.to(device).eval()
    config       = checkpoint_dict['h']
    speaker_list = checkpoint_dict['speakerlist']
    spkr_f0      = checkpoint_dict['speaker_f0_meanstd']
    spkr_sylps   = checkpoint_dict['speaker_sylps_meanstd']
    return model, config, speaker_list, spkr_f0, spkr_sylps

def load_dilated_asr_from_path(checkpoint_path, device='cuda'):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model = init_model_dilated_asr(checkpoint_dict['h'], device=device)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.to(device).eval()
    config       = checkpoint_dict['h']
    speaker_list = checkpoint_dict['speakerlist']
    return model, config, speaker_list

def load_hifigan_ct_from_path(checkpoint_path, device='cuda'):
    model, _, config = load_model_hifigan(checkpoint_path, device=device, return_hparams=True)
    #model.half()
    return model, config

def check_hparams_match(diffsvc_config, dilated_asr_config, hifigan_config):
    important_params = ('n_mel_channels', 'filter_length', 'hop_length', 'win_length', 'mel_fmin', 'mel_fmax', 'n_symbols')
    for param in important_params:
        assert getattr(diffsvc_config, param, None) == getattr(dilated_asr_config, param, None), f'"{param}" param does not match between diffsvc and dilated_asr. Got {getattr(diffsvc_config, param, None)} and {getattr(dilated_asr_config, param, None)} respectively.'
    
    important_vocoder_params = ('n_mel_channels', 'filter_length', 'hop_length', 'win_length', 'mel_fmin', 'mel_fmax')
    for param in important_vocoder_params:
        assert getattr(diffsvc_config, param, None) == getattr(hifigan_config, param, None), f'"{param}" param does not match between diffsvc and hifigan. Got {getattr(diffsvc_config, param, None)} and {getattr(hifigan_config, param, None)} respectively.'

def load_e2e_diffsvc(diffsvc_path, dilated_asr_path, hifigan_path, device='cuda'):
    diffsvc   , diffsvc_config    , speakerlist, spkr_f0, spkr_sylps = load_diffsvc_from_path    (diffsvc_path    , device=device)
    dilatedasr, dilated_asr_config, _                                = load_dilated_asr_from_path(dilated_asr_path, device=device)
    hifigan   , hifigan_config                                       = load_hifigan_ct_from_path (hifigan_path    , device=device)
    check_hparams_match(diffsvc_config, dilated_asr_config, hifigan_config)
    
    speakerlist = [[dataset,name,id,source,source_type,duration] for dataset,name,id,source,source_type,duration in speakerlist if id in spkr_f0.keys()]
    
    stft = get_stft(diffsvc_config)
    
    return diffsvc, dilatedasr, hifigan, stft, diffsvc_config, speakerlist, spkr_f0, spkr_sylps,

def update_loudness(audio, sampling_rate, target_lufs, max_segment_length_s=30.0):
    meter = pyln.Meter(sampling_rate) # create BS.1770 meter
    original_lufs = meter.integrated_loudness(audio[:int(max_segment_length_s*sampling_rate)].numpy()) # measure loudness (in dB)
    original_lufs = torch.tensor(original_lufs).float()
    
    if type(original_lufs) is torch.Tensor:
        original_lufs = original_lufs.to(audio)
    delta_lufs = target_lufs-original_lufs
    gain = 10.0**(delta_lufs/20.0)
    audio = audio*gain
    if audio.abs().max() > 1.0:
        numel_over_limit = (audio.abs() > 1.0).sum()
        if numel_over_limit > audio.numel()/(sampling_rate/16):# if more than 16 samples per second are over 1.0, do peak normalization. Else just clamp them.
            audio /= audio.abs().max()
        audio.clamp_(min=-1.0, max=1.0)
    return audio

def get_audio_from_path(path, config):
    audio, sampling_rate = load_wav_to_torch(path, target_sr=config.sampling_rate)
    audio = update_loudness(audio, sampling_rate, config.target_lufs)
    return audio

def get_mel_from_audio(audio, stft, config):
    mel = stft.mel_spectrogram(audio.detach().cpu().unsqueeze(0))
    return mel

def get_loudness_from_audio(audio, sampling_rate, config):
    meter = pyln.Meter(sampling_rate) # create BS.1770 meter
    lufs_loudness = meter.integrated_loudness(audio[:int(max_segment_length_s*sampling_rate)].numpy()) # measure loudness (in dB)
    lufs_loudness = torch.tensor(lufs_loudness).float()
    return lufs_loudness

def get_pitch(audio, sampling_rate, hop_length, f0_floors=[56.,], f0=None, refine_pitch=True, f0_ceil=1500., voiced_sensitivity=0.13):
    """
    audio: torch.FloatTensor [wav_T]
    sampling_rate: int
    hop_length: int
    f0_floors: list[int]
        - f0_floors is list of minimum pitch values.
          f0 elements of next array replaces previous f0 if elements of previous f0 array are zero
          (aka if the previous f0_floor didn't find any pitch but the next one did, use the next pitch from the next f0_floor)
    """
    if type(f0_floors) in [int, float]:
        f0_floors = [f0_floors,]
    # Extract Pitch/f0 from raw waveform using PyWORLD
    audio = torch.cat((audio, audio[-1:]), dim=0)
    audio = audio.numpy().astype(np.float64)
    
    for f0_floor in f0_floors:
        f0raw, timeaxis = pw.dio(# get raw pitch
            audio, sampling_rate,
            frame_period=(hop_length/sampling_rate)*1000.,# For hop size 256 frame period is 11.6 ms
            f0_floor=f0_floor,# f0_floor : float
                         #     Lower F0 limit in Hz.
                         #     Default: 71.0
            f0_ceil =f0_ceil,# f0_ceil : float
                           #     Upper F0 limit in Hz.
                           #     Default: 800.0
            allowed_range=voiced_sensitivity,# allowed_range : float
                               #     Threshold for voiced/unvoiced decision. Can be any value >= 0, but 0.02 to 0.2
                               #     is a reasonable range. Lower values will cause more frames to be considered
                               #     unvoiced (in the extreme case of `threshold=0`, almost all frames will be unvoiced).
            )
        if refine_pitch:# improves loss values in FastSpeech2 style decoder.
            f0raw = pw.stonemask(audio, f0raw, timeaxis, sampling_rate)# pitch refinement
        f0raw = torch.from_numpy(f0raw).float().clamp(min=0.0, max=f0_ceil)# (Number of Frames) = (654,)
        f0 = f0raw if f0 is None else torch.where(f0==0.0, f0raw, f0)# if current f0 has non-voiced but current f0 has voiced, fill current non-voiced with new voiced pitch.
    voiced_mask = (f0>3)# voice / unvoiced flag
    return f0, voiced_mask# [mel_T], [mel_T]

def get_logf0_from_audio(audio, config):
    f0, vo = get_pitch(audio, config.sampling_rate, config.hop_length, getattr(config, 'f0_floors', [55., 78., 110., 156.]), None, refine_pitch=True, f0_ceil=getattr(config, 'f0_ceil', 1500.), voiced_sensitivity=getattr(config, 'voiced_sensitivity', 0.10))
    logf0 = f0.log().where(vo, f0[0]*0.0)
    return logf0

def get_ppg_from_mel(mel, model, config, mel_lengths=None):
    model_device, model_dtype = next(model.parameters()).device, next(model.parameters()).dtype
    if mel_lengths is None:
        mel_lengths = torch.tensor([mel.shape[2],]).long()# [B, n_mel, mel_T] -> [mel_T]
    ppg = model.generator.align(mel.to(model_device, model_dtype), mel_lengths.to(model_device))
    return ppg

def write_to_file(path, audio, sampling_rate):
    audio = (audio.float() * 2**15).squeeze().cpu().numpy().astype('int16')
    write(path, sampling_rate, audio)

def endtoend_from_path(diffsvc, dilatedasr, hifigan, stft, config, speakerlist, spkr_f0, spkr_sylps,
                       audiopath, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_step_size=1, t_max_step=None):
    audio = get_audio_from_path(audiopath, config)
    pred_audio = endtoend(diffsvc, dilatedasr, hifigan, stft, config, speakerlist, spkr_f0, spkr_sylps,
             audio, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_step_size=t_step_size, t_max_step=t_max_step)
    return pred_audio

def endtoend_from_cache(diffsvc, dilatedasr, hifigan, stft, config, speakerlist, spkr_f0, spkr_sylps,
                       audiopath, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_step_size=1, t_max_step=None):
    audio = get_audio_from_path(audiopath, config)
    pred_audio = endtoend(diffsvc, dilatedasr, hifigan, stft, config, speakerlist, spkr_f0, spkr_sylps,
             audio, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_step_size=t_step_size, t_max_step=t_max_step)
    return pred_audio

@torch.no_grad()
def endtoend(diffsvc, dilatedasr, hifigan, stft, config, speakerlist, spkr_f0, spkr_sylps,
             audio, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_step_size=1, t_max_step=None, gt_mel=None, frame_ppg=None, gt_frame_logf0=None):# only supports a single audio file at a time
    # get input features for model
    if gt_mel is None or frame_ppg is None:
        gt_mel = get_mel_from_audio(audio, stft, config)# [1, n_mel, mel_T]
    if frame_ppg is None:
        frame_ppg = get_ppg_from_mel(gt_mel, dilatedasr, config)
    if gt_frame_logf0 is None:
        gt_frame_logf0 = get_logf0_from_audio(audio, config).unsqueeze(0)
    mel_lengths = torch.tensor([gt_mel.shape[2],]).long()
    gt_perc_loudness = torch.tensor([config.target_lufs,])
    
    # get speaker id from fuzzy speaker name
    possible_names = [x[1].lower() for x in speakerlist]
    speaker_lookup = {x[1].lower(): x[2] for x in speakerlist}
    speaker = difflib.get_close_matches(target_speaker.lower(), possible_names, n=2, cutoff=0.2)[0]# get closest name from target_speaker
    print(f"Selected speaker: {speaker}")
    speaker_id_ext = speaker_lookup[speaker]
    
    # get internal speaker id and pitch/speed characteristics
    (speaker_id,
     speaker_f0_meanstd,
     speaker_slyps_meanstd) = speaker_id_ext, spkr_f0[speaker_id_ext], spkr_sylps[speaker_id_ext]
    speaker_id = torch.tensor([speaker_id,]).long()
    speaker_f0_meanstd    = torch.tensor([speaker_f0_meanstd,])
    speaker_slyps_meanstd = torch.tensor([speaker_slyps_meanstd,])
    
    # (optional) modify pitch to make the speaker sound more like the target
    if correct_pitch:# correct pitch mean
        correction_shift = speaker_f0_meanstd[:, 0].log()-gt_frame_logf0[gt_frame_logf0!=0.0].float().exp().mean().log()
        gt_frame_logf0[gt_frame_logf0!=0.0] += correction_shift
    
    if correct_pitch_logstd:# correct log pitch standard deviation
        target_logstd = (speaker_f0_meanstd[:, 0]+speaker_f0_meanstd[:, 1]).log() - (speaker_f0_meanstd[:, 0]-speaker_f0_meanstd[:, 1]).log()
        mean, current_logstd = gt_frame_logf0[gt_frame_logf0!=0.0].mean(), gt_frame_logf0[gt_frame_logf0!=0.0].std()
        correction_scale = target_logstd/current_logstd
        gt_frame_logf0[gt_frame_logf0!=0.0] = gt_frame_logf0[gt_frame_logf0!=0.0].sub(mean).mul(correction_scale).add(mean)
    
    if correct_pitch_std:# correct pitch standard deviation
        correction_scale = speaker_f0_meanstd[:, 1]/gt_frame_logf0[gt_frame_logf0!=0.0].float().exp().std()
        mean = gt_frame_logf0[gt_frame_logf0!=0.0].mean()
        gt_frame_logf0[gt_frame_logf0!=0.0] = gt_frame_logf0[gt_frame_logf0!=0.0].sub(mean).exp().mul(correction_scale).log().add(mean)
    
    # move all features to correct device + dtype
    diffsvc_device, diff_dtype = next(diffsvc.parameters()).device, next(diffsvc.parameters()).dtype
    gt_mel                = gt_mel               .to(diffsvc_device, diff_dtype)
    gt_perc_loudness      = gt_perc_loudness     .to(diffsvc_device, diff_dtype)
    gt_frame_logf0        = gt_frame_logf0       .to(diffsvc_device, diff_dtype)
    frame_ppg             = frame_ppg            .to(diffsvc_device, diff_dtype)
    mel_lengths           = mel_lengths          .to(diffsvc_device, torch.long)
    speaker_id            = speaker_id           .to(diffsvc_device, torch.long)
    speaker_f0_meanstd    = speaker_f0_meanstd   .to(diffsvc_device, diff_dtype)
    speaker_slyps_meanstd = speaker_slyps_meanstd.to(diffsvc_device, diff_dtype)
    
    # run DiffSVC over inputs to get spectrogram
    pred_mel = diffsvc.generator.voice_conversion_main(
                       gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B] # take from reference/source
                           gt_perc_loudness,# FloatTensor[B]                              # take from reference/source
                             gt_frame_logf0,# FloatTensor[B, mel_T]                       # take from reference/source
                                  frame_ppg,# FloatTensor[B, ppg_dim, mel_T]              # take from reference/source
                                 speaker_id,#  LongTensor[B]                              # take from target speaker
                         speaker_f0_meanstd,# FloatTensor[B, 2]                           # take from target speaker
                      speaker_slyps_meanstd,# FloatTensor[B, 2]                           # take from target speaker
                              t_step_size=t_step_size,# int
                               t_max_step=t_max_step).transpose(1, 2)# -> [B, n_mel, mel_T]
    
    # run HiFi-GAN over spectrogram to get audio
    hifigan_device, hifigan_dtype = next(hifigan.parameters()).device, next(hifigan.parameters()).dtype
    pred_audio = hifigan(pred_mel.to(hifigan_device, hifigan_dtype))
    
    return pred_audio



# testing
def test_wav():# test the model with data computed from the functions above
    outdir = "/media/cookie/Samsung 860 QVO/TTS/"
    audiopath = "/media/cookie/Samsung 860 QVO/TTS/voice_sample.wav"
    
    target_speakers = ['Discord','Twilight','Pinkie','Nancy','Yosuke','Adachi']
    correct_pitch = True
    correct_pitch_logstd = False
    correct_pitch_std    = True
    device = 'cpu'
    
    diffsvc, dilatedasr, hifigan, stft, diffsvc_config, speakerlist, spkr_f0, spkr_sylps, = load_e2e_diffsvc(
        diffsvc_path     = "/media/cookie/Samsung PM961/TwiBot/CookiePPPTTS/CookieTTS/experiments/DiffSVC/outdir_015_7x3/latest_val_model",
        dilated_asr_path = "/media/cookie/Samsung PM961/TwiBot/CookiePPPTTS/CookieTTS/experiments/dilated_ASR/outdir_002/checkpoint_87000",
        hifigan_path     = "/media/cookie/Samsung PM961/TwiBot/CookiePPPTTS/CookieTTS/_4_mtw/hifigan_ct/outdir_u4_warm_oggless/latest_val_model",
        device=device,
    )
    
    lin_start   = 1e-4                                    # default 1e-4
    lin_end     = 0.24# modify for more strength per step # default 0.06
    lin_n_steps = 200 # modify for more steps             # default 100
    diffsvc.generator.diffusion.set_noise_schedule(lin_start, lin_end, lin_n_steps, device=device)
    
    for target_speaker in target_speakers:
        for max_t in range(0, lin_n_steps+1, lin_n_steps//2):
            start_time = time.time()
            pred_audio = endtoend_from_path(diffsvc, dilatedasr, hifigan, stft, diffsvc_config, speakerlist, spkr_f0, spkr_sylps,
                                            audiopath, target_speaker, correct_pitch, correct_pitch_logstd, correct_pitch_std, t_max_step=max_t)
            
            outpath = f"{outdir}/output_spkr{target_speaker}_max{max_t:04}_{'mod' if correct_pitch else 'orig'}pitch.wav"
            write_to_file(outpath, pred_audio, diffsvc_config.sampling_rate)
            print(f"Wrote audio to '{outpath}'\nTook {time.time()-start_time:.2f} seconds")
            print("")

if __name__ == "__main__":
    test_wav()

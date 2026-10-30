# Datasets

The datasets are not included in this repository. Download them from the official sources below, accept their licences, and place them under `datasets/` in the repository root. `code/prepare_data.py` searches these folders recursively, so the exact sub-folder names inside each dataset folder do not matter, as long as the files listed below are present.

| Dataset | Official source | Licence | Used files |
|---|---|---|---|
| MELD | https://affective-meld.github.io/ | see the MELD repository (GPL-3.0) | `MELD.Raw` videos and the `train/dev/test_sent_emo.csv` label files |
| CREMA-D | https://github.com/CheyneyComputerScience/CREMA-D | Open Database License (ODbL) | `VideoFlash/*.flv` (video) and `AudioWAV/*.wav` (audio) |
| RAVDESS | https://doi.org/10.5281/zenodo.1188976 | CC BY-NC-SA 4.0 | `Video_Speech_Actor_01.zip` … `Video_Speech_Actor_24.zip` (audio-visual speech, modality code 01) |

## Expected layout

```
datasets/
  MELD/
    ...  train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
    ...  train_splits/              dia*_utt*.mp4
    ...  dev_splits_complete/       dia*_utt*.mp4
    ...  output_repeated_splits_test/ dia*_utt*.mp4
  CREMA-D/
    VideoFlash/   1001_DFA_ANG_XX.flv ...
    AudioWAV/     1001_DFA_ANG_XX.wav ...
  RAVDESS/
    Video_Speech_Actor_01/Actor_01/01-01-01-01-01-01-01.mp4 ...
    ...
    Video_Speech_Actor_24/Actor_24/...
```

## Statistics after preprocessing

| Dataset | Clips | Speakers | Classes | Protocol |
|---|---|---|---|---|
| MELD | 13,706 | 304 | 7 | official train / dev / test, 3 seeds |
| CREMA-D | 7,442 | 91 | 6 | speaker-independent 5-fold CV |
| RAVDESS | 1,440 | 24 | 8 | speaker-independent 5-fold CV |

Please cite the original dataset papers when using them:

- S. Poria et al., MELD: A multimodal multi-party dataset for emotion recognition in conversations, ACL 2019. https://doi.org/10.18653/v1/P19-1050
- H. Cao et al., CREMA-D: Crowd-sourced emotional multimodal actors dataset, IEEE Trans. Affective Computing 5(4), 2014. https://doi.org/10.1109/TAFFC.2014.2336244
- S. R. Livingstone, F. A. Russo, The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), PLoS ONE 13(5), 2018. https://doi.org/10.1371/journal.pone.0196391

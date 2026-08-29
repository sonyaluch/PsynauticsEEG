# PsynauticsEEG
EEG analysis for Psynautics at-home study

Resting-state EEG collected with Muse S (Gen 2)
headbands as part of the Psynautics protocol: an 11-day citizen-neuroscience
study with baseline (Days 1-5), treatment (Day 6), and post-treatment (Days
7-11) recordings, 4 channels (TP9, AF7, AF8, TP10).

The end goal is three derived markers per recording -- **Cognition** (AF8
frontal spectral power), **Emotion** (AF7/AF8 alpha asymmetry), and
**Awareness** (Lempel-Ziv complexity / permutation entropy across all 4
channels). As a first step, preprocessing covers loading, cleaning, and QC-flagging raw exports into ready-to-analyze MNE `Raw` objects.

# Third-party notices

VoiceRail re-implements published model architectures so that it can load
their public weights. The original projects and their licenses:

| Component | Used for | Upstream | License |
|---|---|---|---|
| RVC (Retrieval-based Voice Conversion) | Synthesizer architecture, inference pipeline design, pretrained v2 weights | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT |
| VITS | Text encoder, flow and posterior encoder | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) | MIT |
| HiFi-GAN / NSF | Vocoder and discriminators | [jik876/hifi-gan](https://github.com/jik876/hifi-gan), [nii-yamagishilab/project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) | MIT / BSD-3 |
| RMVPE | Pitch estimation network and weights | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) (via RVC) | MIT |
| ContentVec / HuBERT | Content feature encoder weights | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec), [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) | MIT |
| audio-slicer | Silence slicing algorithm | [openvpi/audio-slicer](https://github.com/openvpi/audio-slicer) | MIT |
| Applio / Ultimate RVC | Reference implementations consulted for compatibility | [IAHispano/Applio](https://github.com/IAHispano/Applio), [JackismyShephard/ultimate-rvc](https://github.com/JackismyShephard/ultimate-rvc) | MIT |
| TITAN pretrained models | Optional training base | [blaise-tk/TITAN](https://huggingface.co/blaise-tk/TITAN) | see model card |

Separation models (Mel-/BS-RoFormer, MDX-Net and friends) are downloaded at
runtime through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT); each model is the work of its respective author (Kimberley Jensen,
unwa, ViperX, aufr33, anvuew, Sucial, becruily, the UVR team and others).

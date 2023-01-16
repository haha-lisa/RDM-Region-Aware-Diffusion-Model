# Region-Aware Diffusion for Zero-shot Text-driven Image Editing

This is the official PyTorch implementation of the paper ''Region-Aware Diffusion for Zero-shot Text-driven Image Editing''.

![MAIN3_e2-min](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model/blob/main/teaser.png)

## Abstract
Image manipulation under the guidance of textual descriptions has recently received a broad range of attention. In this study, we focus on the regional editing of images with the guidance of given text prompts. Different from current mask-based image editing methods, we propose a novel region-aware diffusion model (RDM) for entity-level image editing, which could automatically locate the region of interest and replace it following given text prompts. To strike a balance between image fidelity and inference speed, we design the intensive diffusion pipeline by combing latent space diffusion and enhanced directional guidance. In addition, to preserve image content in non-edited regions, we introduce regional-aware entity editing to modify the region of interest and preserve the out-of-interest region. We validate the proposed RDM beyond the baseline methods through extensive qualitative and quantitative experiments. The results show that RDM outperforms the previous approaches in terms of visual quality, overall harmonization, non-editing region content preservation, and text-image semantic consistency.

## Framework
![MAIN3_e2-min](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model/blob/main/pipeline.png)
The overall framework of RDM.


## Install dependencies
```
git clone https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model
cd RDM-Region-Aware-Diffusion-Model
install [latent diffusion](https://github.com/CompVis/latent-diffusion)
pip install -e .
```

## Pretrained models
[bert](https://dall-3.com/models/glid-3-xl/bert.pt), [kl-f8](https://dall-3.com/models/glid-3-xl/kl-f8.pt), [diffusion](https://dall-3.com/models/glid-3-xl/inpaint.pt)
<br> Please download them and put them into the floder ./ <br> 


## Run
```
python run_edit.py --edit ./input_image/flower1.jpg --mask ./input_image/flower1_mask.png \
-fp "a flower" --batch_size 6 --num_batches 2 \
--text "a chrysanthemum" --prefix "test_flower"
```

## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file.<br>
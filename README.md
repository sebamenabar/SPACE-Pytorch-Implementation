# SPACE-Pytorch-Implementation
SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition https://arxiv.org/abs/2001.02407

Note: This is a highly opinionated (by me) implementation from what I can understand from the paper and makes sense to me.

## TODO
* [ ] Foreground inference
    * [x] Image Encoder
    * [x] $ZNet$
    * [x] $Z$ Sampling
    * [x] Construct $z_{where}$
    * [x] Extract glimpses with a Spatial Transformer
    * [x] Glimpse Encoder
    * [ ] Glimpse Decoder
      * [ ] Foreground mask and appearance
    * [ ] Compute weights for each component
    * [ ] Compute global weighted mask and foreground appearance
* [ ] Background Inference
  * [ ] Image Encoder
  * [ ] Predict Mask
  * [ ] Decode Mask
  * [ ] Stick breaking process as described in GENESIS
  * [ ] Component Encoder
  * [ ] Component Decoder
* [ ] Background Generation
  * [ ] Everything
* [ ] Renderer (?)
* [ ] Loss
  * [ ] ELBO Loss
    * [ ] Reconstruction Error
    * [ ] KL Divergence
  * [ ] Boundary Loss

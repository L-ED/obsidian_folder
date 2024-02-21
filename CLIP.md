
Idea 
learn from text embedding as supervision. 

How to learn 
Transformer language model as text encoder too slow. Bag of words prediction is faster. SSL even faster 

Starting with the same bag-of-words encoding baseline, we swapped the predictive objective for a contrastive objective in Figure 2 and observed a further 4x efficiency improvement in the rate of zero-shot transfer to ImageNet.

Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible (image, text) pairings across a batch actually occurred. To do this, CLIP  to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N*N − N incorrect pairings. We optimize a symmetric cross entropy loss over these similarity scores

```
# image_encoder - ResNet or Vision Transformer 
# text_encoder - CBOW or Text Transformer 
# I[n, h, w, c] - minibatch of aligned images 
# T[n, l] - minibatch of aligned texts 
# W_i[d_i, d_e] - learned proj of image to embed 
# W_t[d_t, d_e] - learned proj of text to embed 
# t - learned temperature parameter 

# extract feature representations of each modality 
I_f = image_encoder(I) #[n, d_i] 
T_f = text_encoder(T) #[n, d_t] 

# joint multimodal embedding [n, d_e] 
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 

# scaled pairwise cosine similarities [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t) 

# symmetric loss function 
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t)/2
```

To our knowledge this batch construction technique and objective was first introduced in the area of deep metric learning as the multi-class N-pair loss Sohn (2016), was popularized for contrastive representation learning by Oord et al. (2018) as the InfoNCE loss, and was recently adapted for contrastive (text, image) representation learning in the domain of medical imaging by Zhang et al. (2020).

We instead use only a linear projection to map from each encoder’s representation to the multi-modal embedding space. We also remove the text transformation function tu from Zhang et al. (2020) which samples a single sentence at uniform from the text since many of the (image, text) pairs in CLIP’s pretraining dataset are only a single sentence. We also simplify the image transformation function tv. A random square crop from resized images is the only data augmentation used during training.

Architecture

Image encoder
ResNet-50 (He et al., 2016a) as the base with several modifications:
1. using the ResNetD improvements from He et al. (2019) 
2. antialiased rect-2 blur pooling from Zhang (2019). 
3. Replace the global average pooling layer with an attention pooling mechanism as a single layer of “transformer-style” multi-head QKV attention where the query is conditioned on the global average-pooled representation of the image. 

For the second architecture, we experiment with the recently introduced Vision Transformer (ViT) (Dosovitskiy et al., 2020). We closely follow their implementation with only the minor modification of adding an additional layer normalization to the combined patch and position embeddings before the transformer and use a slightly different initialization scheme

Text encoder 
The text encoder is a Transformer (Vaswani et al., 2017) with modifications described in Radford et al. (2019). As a base size we use a 63M-parameter 12- layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size (Sennrich et al., 2015). For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space. Masked self-attention was used in the text encoder to preserve the ability to initialize with a pre-trained language model or add language modeling as an auxiliary objective, though exploration of this is left as future work.

Creating sentences with target word improves acc
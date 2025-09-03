# EviD-GAN: Improving GAN With an Infinite Set of Discriminators at Negligible Cost

### Abstract :

>Ensemble learning improves the capability of convolutional neural network (CNN)-based discriminators, whose performance is crucial to the quality of generated samples in generative adversarial network (GAN). However, this learning strategy results in a significant increase in the number of parameters along with computational overhead. Meanwhile, the suitable number of discriminators required to enhance GAN performance is still being investigated. To mitigate these issues, we propose an evidential discriminator for GAN (EviD-GAN) to learn both the model (epistemic) and data (aleatoric) uncertainties. Specifically, by analyzing three GAN models, the relation between the distribution of discriminator’s output and the generator performance has been discovered yielding a general formulation of GAN framework. With the above analysis, the evidential discriminator learns the degree of aleatoric and epistemic uncertainties via imposing a higher order distribution constraint over the likelihood as expressed in the discriminator’s output. This constraint can learn an ensemble of likelihood functions corresponding to an infinite set of discriminators. Thus, EviD-GAN aggregates knowledge through the ensemble learning of discriminator that allows the generator to benefit from an informative gradient flow at a negligible computational cost. Furthermore, inspired by the gradient direction in maximum mean discrepancy (MMD)-repulsive GAN, we design an asymmetric regularization scheme for EviD-GAN. Unlike MMD-repulsive GAN that performs at the distribution level, our regularization scheme is based on a pairwise loss function, performs at the sample level, and is characterized by an asymmetric behavior during the training of generator and discriminator. Experimental results show that the proposed evidential discriminator is cost-effective, consistently improves GAN in terms of Frechet inception distance (FID) and inception score (IS), and performs better than other competing models that use multiple discriminators.

## 📜  License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute EviD-GAN in either commercial or academic projects under the terms of this license.

## 📚 Citation

If you use **EviD-GAN** in your research or applications, please consider citing our paper:

```bibtex
@ARTICLE{10639254,
  author={Gnanha, Aurele Tohokantche and Cao, Wenming and Mao, Xudong and Wu, Si and Wong, Hau-San and Li, Qing},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={EviD-GAN: Improving GAN With an Infinite Set of Discriminators at Negligible Cost}, 
  year={2025},
  volume={36},
  number={4},
  pages={6422-6436},
  keywords={Generative adversarial networks;Training;Uncertainty;Generators;Computational modeling;Data models;Ensemble learning;Deep learning;evidential learning;generative adversarial networks (GANs);generative modeling},
  doi={10.1109/TNNLS.2024.3388197}}
```

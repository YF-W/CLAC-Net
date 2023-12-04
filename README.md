# CLAC-Net
 Improving Accuracy in the Analysis of Medical Images via Cross-Layer Fusion and Asymmetric Connections

# Description
The Cross-layer and Asymmetric-connection Net (CLAC-Net) represents a significant advancement in medical image semantic segmentation, crucial for identifying organs and lesions, evaluating pathologies, and aiding in surgical procedures. This innovative network model tackles common challenges faced by existing models, such as:

1. The struggle to effectively synchronize and integrate contextual semantic information.
2. The limited capability in assimilating global data with local context.
3. The semantic inconsistencies arising from upsampling and downsampling processes.

CLAC-Net is an evolution of the ACUNet framework, incorporating strategic design elements:

a. It features unique asymmetric skip connections, specifically engineered to minimize the loss of features across extended sequences. This enhances the alignment of semantic information across different contexts.

b. The model introduces the Cross-Layer Relay Knot System (CLRKS). This system is pivotal in gathering and fusing global insights with detailed contextual information.

c. A key component of CLAC-Net is the Deep Embedding Attention Bottleneck (DEAB). This element is crucial in fortifying the semantic link between the encoder and decoder portions of the model, effectively bridging the semantic divide.

Through rigorous testing, CLAC-Net has demonstrated its effectiveness, outperforming similar models in various comparative studies. This makes it a valuable tool in the field of medical imaging and diagnostics.

# Baseline of CLAC-Net
<img width="698" alt="baseline" src="https://github.com/YF-W/CLAC-Net/assets/66008255/93afd068-a06b-4c5e-b0fc-cb303c5170c9">


# Architecture of CLAC-Net
<img width="863" alt="CLAC" src="https://github.com/YF-W/CLAC-Net/assets/66008255/998f86fd-09b2-44ea-b31e-6b86da5f24de">

# CLAC_structure
![CLAC_structure](https://github.com/YF-W/CLAC-Net/assets/66008255/2d4b5132-8ae8-4d6a-a108-365df48db4b9)


# **Environment**

IDE: PyCharm 2020.1 Professional Edition.

Framework:  PyTorch 1.13.0.

Language: Python 3.8.15

CUDA: 11.7

# **Awesome NLP Papers**

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/user/Awesome-NLP)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

---

**Awesome NLP** is a curated collection of high-quality resources, papers, libraries, tools, and datasets for **Natural Language Processing (NLP)**. Whether you're a beginner exploring the basics or an expert diving into cutting-edge research, this repository has something for everyone.

---

## **Contents**

- [1. Introduction](#1-introduction)
- [2. How to Use](#2-how-to-use)
- [3. Contributing](#3-contributing)
- [4. Fundamentals of Deep Learning](#4-fundamentals-of-deep-learning)
  - [4.1 Neural Networks and Deep Learning](#41-neural-networks-and-deep-learning)
  - [4.2 Activation Functions](#42-activation-functions)
  - [4.3 Backpropagation and Gradient Descent](#43-backpropagation-and-gradient-descent)
  - [4.4 Optimization Techniques](#44-optimization-techniques)
- [5. Sequence Modeling](#5-sequence-modeling)
  - [5.1 RNNs and LSTMs](#51-rnns-and-lstms)
  - [5.2 Sequence Models](#52-sequence-models)
  - [5.3 Attention Mechanism](#53-attention-mechanism)
  - [5.4 Transformers](#54-transformers)
- [6. Word Representations](#6-word-representations)
  - [6.1 Static Word Embeddings](#61-static-word-embeddings)
  - [6.2 Contextualized Embeddings](#62-contextualized-embeddings)
  - [6.3 Subword-Based Representations](#63-subword-based-representations)
- [7. Evaluation](#7-evaluation)
  - [7.1 Evaluation Metrics (Accuracy, BLEU, ROUGE, etc.)](#71-evaluation-metrics-accuracy-bleu-rouge-etc)
  - [7.2 Model Validation and Cross-validation](#72-model-validation-and-cross-validation)
  - [7.3 Bias and Fairness Metrics](#73-bias-and-fairness-metrics)
- [8. Tasks](#8-tasks)
  - [8.1 Text Generation](#81-text-generation)
  - [8.2 Text Classification](#82-text-classification)
  - [8.3 Named Entity Recognition](#83-named-entity-recognition)
  - [8.4 Question Answering](#84-question-answering)
  - [8.5 Fill Mask](#85-fill-mask)
  - [8.6 Machine Translation](#86-machine-translation)
- [9. Models](#9-models)
  - [9.1 BERT](#91-bert)
  - [9.2 GPT-3 (GPT)](#92-gpt-3-gpt)
  - [9.3 GPT-2](#93-gpt-2)
  - [9.4 RoBERTa](#94-roberta)
  - [9.5 T5](#95-t5)
  - [9.6 DistilBERT](#96-distilbert)
  - [9.7 ALBERT](#97-albert)
  - [9.8 BART](#98-bart)
  - [9.9 ELECTRA](#99-electra)
  - [9.10 XLNet](#910-xlnet)
  - [9.11 BERTweet](#911-berttweet)
  - [9.12 BlenderBot](#912-blenderbot)
  - [9.13 DeBERTa](#913-deberta)
  - [9.14 BigBird](#914-bigbird)
  - [9.15 PEGASUS](#915-pegasus)
  - [9.16 FLAN-T5](#916-flan-t5)
  - [9.17 MobileBERT](#917-mobilebert)
  - [9.18 GPT-Neo](#918-gpt-neo)
  - [9.19 Longformer](#919-longformer)
  - [9.20 XLM-RoBERTa](#920-xlm-roberta)
  - [9.21 DialoGPT](#921-dialogpt)
  - [9.22 MarianMT](#922-marianmt)
  - [9.23 Falcon](#923-falcon)
  - [9.24 CodeGen](#924-codegen)
  - [9.25 ByT5](#925-byt5)
  - [9.26 PhoBERT](#926-phobert)
  - [9.27 Funnel Transformer](#927-funnel-transformer)
  - [9.28 T5v1.1](#928-t5v1-1)
  - [9.29 RoFormer](#929-roformer)
  - [9.30 MBart and MBart-50](#930-mbart-and-mbart-50)
- [10. Datasets](#10-datasets)
  - [10.1 Text Generation Datasets](#101-text-generation-datasets)
  - [10.2 Text Classification Datasets](#102-text-classification-datasets)
  - [10.3 Named Entity Recognition Datasets](#103-named-entity-recognition-datasets)
  - [10.4 Question Answering Datasets](#104-question-answering-datasets)
  - [10.5 Fill Mask Datasets](#105-fill-mask-datasets)
  - [10.6 Machine Translation Datasets](#106-machine-translation-datasets)
- [11. NLP in Vietnamese](#11-nlp-in-vietnamese)
  - [11.1 Vietnamese Text Preprocessing](#111-vietnamese-text-preprocessing)
  - [11.2 Vietnamese Word Representations](#112-vietnamese-word-representations)
  - [11.3 Vietnamese Named Entity Recognition (NER)](#113-vietnamese-named-entity-recognition-ner)
  - [11.4 Vietnamese Part-of-Speech Tagging](#114-vietnamese-part-of-speech-tagging)
  - [11.5 Vietnamese Syntax and Parsing](#115-vietnamese-syntax-and-parsing)
  - [11.7 Machine Translation for Vietnamese](#116-machine-translation-for-vietnamese)
  - [11.8 Vietnamese Question Answering](#117-vietnamese-question-answering)
  - [11.9 Vietnamese Text Summarization](#118-vietnamese-text-summarization)
  - [11.10 Resources for Vietnamese NLP](#119-resources-for-vietnamese-nlp)
  - [11.11 Challenges in Vietnamese NLP](#1110-challenges-in-vietnamese-nlp)

---

## **1. Introduction**

Natural Language Processing (NLP) is a fast-evolving field at the intersection of **linguistics, artificial intelligence, and deep learning**. It powers various applications, from **chatbots and machine translation** to **automated text generation and information retrieval**.

This repository organizes NLP research into key areas, making it easier for students, researchers, and practitioners to find relevant papers, tools, and datasets. Below is an overview of the main sections:

- **[Fundamentals of Deep Learning](#4-fundamentals-of-deep-learning)**: Covers the **core concepts** of deep learning, including neural networks, activation functions, backpropagation, and optimization techniques.
- **[Sequence Modeling](#5-sequence-modeling)**: Focuses on **sequential data processing**, including Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Transformer-based architectures.

- **[Word Representations](#6-word-representations)**: Explores **word embedding techniques**, including **static embeddings** (Word2Vec, GloVe) and **contextualized embeddings** (BERT, ELMo).

- **[Evaluation](#7-evaluation)**: Discusses **how to measure NLP model performance**, including accuracy, BLEU, ROUGE, and fairness metrics.

- **[Tasks](#8-tasks)**: A collection of research papers on key NLP applications such as **text generation, classification, named entity recognition (NER), question answering, and machine translation**.

- **[Models](#9-models)**: Covers state-of-the-art NLP models such as **BERT, GPT-3, RoBERTa, T5, and many others**, providing links to research papers and implementations.

- **[Datasets](#10-datasets)**: A list of **public datasets** commonly used in NLP research, categorized by task (e.g., text classification, NER, machine translation).

- **[NLP in Vietnamese](#11-nlp-in-vietnamese)**: Focuses on **Vietnamese NLP research**, including text preprocessing, embeddings, sentiment analysis, and translation.

This structured collection makes it easier to **understand fundamental NLP concepts**, **explore the latest research**, and **apply NLP techniques to real-world problems**.

---

## **2. How to Use**

This repository is designed to be a **comprehensive reference** for NLP research and applications. Here’s how you can make the most of it:

### 1️⃣ **Learn the Basics**

If you're new to NLP, start with the **[Fundamentals of Deep Learning](#4-fundamentals-of-deep-learning)** section. It provides a foundation in deep learning concepts that are essential for understanding modern NLP techniques.

### 2️⃣ **Explore NLP Architectures**

Read about different **sequence modeling techniques** in the **[Sequence Modeling](#5-sequence-modeling)** section. This will introduce you to RNNs, LSTMs, the Attention Mechanism, and the Transformer model, which forms the basis of most modern NLP models.

### 3️⃣ **Understand Word Representations**

Check out the **[Word Representations](#6-word-representations)** section to learn how text is transformed into numerical vectors, including **static embeddings (Word2Vec, GloVe)** and **contextualized embeddings (BERT, ELMo, GPT)**.

### 4️⃣ **Assess Model Performance**

Visit the **[Evaluation](#7-evaluation)** section to understand how NLP models are evaluated. This section covers **common metrics** such as BLEU for translation, ROUGE for summarization, and fairness metrics.

### 5️⃣ **Find NLP Research Papers by Task**

Browse the **[Tasks](#8-tasks)** section for papers related to **text classification, question answering, machine translation, and more**.

### 6️⃣ **Explore State-of-the-Art NLP Models**

Visit the **[Models](#9-models)** section to find research papers on models like **BERT, GPT-3, RoBERTa, T5**, and others.

### 7️⃣ **Discover NLP Datasets**

If you're looking for training datasets, check out the **[Datasets](#10-datasets)** section, which categorizes datasets based on NLP tasks.

### 8️⃣ **Explore Vietnamese NLP Research**

For researchers focusing on Vietnamese NLP, the **[NLP in Vietnamese](#11-nlp-in-vietnamese)** section includes papers and resources on **Vietnamese text preprocessing, NER, sentiment analysis, and machine translation**.

### 9️⃣ **Stay Updated**

The field of NLP is evolving rapidly. Keep an eye on new research papers and updates to this repository.

### 🔟 **Contribute and Collaborate**

If you have found a useful NLP paper or tool, consider contributing! See the [Contributing](#contributing) section for details.

---

## **3. Contributing**

We welcome contributions to make this repository better! Here’s how you can help:

1. **Suggest Papers or Resources:**  
   Found an important NLP paper, dataset, or tool? Open an **issue** or submit a **pull request**.

2. **Report Issues:**  
   Noticed a broken link or incorrect information? Let us know by opening an issue.

3. **Enhance Documentation:**  
   Help improve descriptions, summaries, or structure.

4. **Submit Pull Requests:**
   - **Fork** the repository.
   - **Create a new branch** for your changes.
   - **Commit your updates**, ensuring they follow the existing format.
   - **Submit a pull request** with a clear description of your contribution.

Check out our [Contribution Guidelines](CONTRIBUTING.md) for detailed instructions.

---

## **Next Steps**

- If you find this repository helpful, **star ⭐ it on GitHub** and **share it with the NLP community**.
- Start exploring topics from the **table of contents**.
- Feel free to contribute by adding **new papers, tools, or datasets**.

---

Happy Learning! 🚀

## **4. Fundamentals of Deep Learning**

This section covers the foundational concepts of deep learning, including neural networks, activation functions, backpropagation, gradient descent, and optimization techniques. Each subsection includes links to important research papers and descriptions for further reading.

### **4.1 Neural Networks and Deep Learning**

Explore the fundamental building blocks of deep learning and their applications across various domains.

- [**Deep cybersecurity: a comprehensive overview from neural network and deep learning perspective**](https://link.springer.com/article/10.1007/s42979-021-00535-6)<br />
  _Authors_: Sarker, Iqbal H<br />
  _Description_: This paper provides an in-depth exploration of neural networks and deep learning applications in cybersecurity. It discusses frameworks, challenges, and future research directions, emphasizing adaptability in cyber defense.

- [**Introduction to machine learning, neural networks, and deep learning**](https://tvst.arvojournals.org/article.aspx?articleid=2762344)<br />
  _Authors_: Choi, Rene Y; Coyner, Aaron S; Kalpathy-Cramer, Jayashree; Chiang, Michael F; Campbell, J Peter<br />
  _Description_: A foundational overview of machine learning principles, focusing on neural networks and deep learning methodologies applied in medical imaging and diagnostics.

- [**An introduction to neural networks and deep learning**](https://www.sciencedirect.com/science/article/abs/pii/B978012810408800002X)<br />
  _Authors_: Suk, Heung-Il<br />
  _Description_: A comprehensive introduction to neural network structures and their progression into deep learning systems, focusing on practical medical applications.

- [**Survey on neural network architectures with deep learning**](https://www.academia.edu/download/70861228/06.pdf)<br />
  _Authors_: Smys, S; Chen, J; Shakya, S<br />
  _Description_: A taxonomy of neural network architectures and their design paradigms, highlighting optimization techniques and use cases across industries.

- [**Fundamentals of artificial neural networks and deep learning**](https://link.springer.com/chapter/10.1007/978-3-030-89010-0_10)<br />
  _Authors_: Montesinos López, O. A; Montesinos López, A<br />
  _Description_: A theoretical exploration of artificial neural networks, detailing their evolution into advanced deep learning systems.

- [**Conceptual understanding of convolutional neural network: A deep learning approach**](https://www.sciencedirect.com/science/article/pii/S1877050918308019)<br />
  _Authors_: Indolia, S; Goswami, A K; Asopa, P<br />
  _Description_: Insights into CNNs as a cornerstone of deep learning, showcasing their advantages for high-dimensional data.

- [**Application of meta-heuristic algorithms for training neural networks and deep learning architectures**](https://link.springer.com/article/10.1007/s11063-022-11055-6)<br />
  _Authors_: Kaveh, M; Mesgari, M S<br />
  _Description_: A review of optimization algorithms applied to neural networks, emphasizing hyperparameter tuning and performance enhancement.

- [**Neural networks and deep learning in urban geography: A systematic review and meta-analysis**](https://www.sciencedirect.com/science/article/pii/S0198971518302928)<br />
  _Authors_: Grekousis, G<br />
  _Description_: An analysis of deep learning applications in urban studies, offering insights into spatial modeling using neural networks.

- [**Deep learning neural networks: Design and case studies**](https://books.google.com/books?hl=en&lr=&id=e5hIDQAAQBAJ&oi=fnd&pg=PR7&dq=Neural+Networks+and+Deep+Learning)<br />
  _Authors_: Graupe, D<br />
  _Description_: A textbook exploring neural network design, training methods, and real-world case studies.

- [**Deep learning in neural networks: An overview**](https://www.sciencedirect.com/science/article/pii/S0893608014002135)<br />
  _Authors_: Schmidhuber, J<br />
  _Description_: A highly cited review covering the history, methodologies, and applications of deep learning.

---

### **4.2 Activation Functions**

Learn about the key role of activation functions in neural networks and their impact on model performance.

- [**A Universal Activation Function for Deep Learning**](https://cdn.techscience.cn/files/cmc/2023/TSP_CMC-75-2/TSP_CMC_37028/TSP_CMC_37028.pdf)<br />
  _Authors_: Hwang, S. Y. & Kim, J. J.<br />
  _Description_: Proposes a novel activation function adaptable across tasks, enhancing model performance and reducing training complexity.

- [**Enhancing Brain Tumor Detection: A Novel CNN Approach with Advanced Activation Functions**](https://pmc.ncbi.nlm.nih.gov/articles/PMC11449684/)<br />
  _Authors_: Kaifi, R.<br />
  _Description_: Develops a specialized activation function tailored for medical imaging, significantly improving accuracy in tumor detection.

- [**An Overview of the Activation Functions Used in Deep Learning Algorithms**](https://dergipark.org.tr/en/doi/10.54187/jnrs.1011739)<br />
  _Authors_: Kılıçarslan, S., Adem, K., & Çelik, M.<br />
  _Description_: Reviews a broad spectrum of fixed and trainable activation functions, discussing their computational properties and impacts.

- [**Smish: A Novel Activation Function for Deep Learning Methods**](https://www.mdpi.com/2079-9292/11/4/540/pdf)<br />
  _Authors_: Wang, X., Ren, H., & Wang, A.<br />
  _Description_: Introduces 'Smish,' a smooth, non-monotonic activation function that outperforms traditional functions in various scenarios.

- [**Learning Specialized Activation Functions for Physics-Informed Neural Networks**](https://arxiv.org/pdf/2308.04073)<br />
  _Authors_: Wang, H., Lu, L., Song, S., & Huang, G.<br />
  _Description_: Focuses on customized activation functions designed for solving physics-informed problems with neural networks.

- [**Rmaf: ReLU-Memristor-like Activation Function for Deep Learning**](https://ieeexplore.ieee.org/abstract/document/9066950/)<br />
  _Authors_: Yu, Y., Adu, K., & Wang, X.<br />
  _Description_: Proposes an activation function inspired by memristive properties to enhance network flexibility and learning.

- [**Catalysis of Neural Activation Functions: Adaptive Feed-forward Training for Big Data Applications**](https://link.springer.com/article/10.1007/s10489-021-03082-y)<br />
  _Authors_: Sarkar, S., Agrawal, S., & Baker, T.<br />
  _Description_: Explores dynamic activation functions that adapt during training, optimizing performance for large-scale datasets.

- [**The Most Used Activation Functions: Classic Versus Current**](https://www.researchgate.net/profile/Marina-Mercioni-2/publication/341958919_The_Most_Used_Activation_Functions_Classic_Versus_Current/links/64e479950acf2e2b52099552/The-Most-Used-Activation-Functions-Classic-Versus-Current.pdf)<br />
  _Authors_: Mercioni, M. A., & Holban, S.<br />
  _Description_: Compares traditional and modern activation functions, identifying trends and shifts in their usage.

- [**Deep Learning Activation Functions: Fixed-Shape, Parametric, Adaptive, Stochastic**](https://arxiv.org/pdf/2407.11090)<br />
  _Authors_: Hammad, M. M.<br />
  _Description_: Categorizes activation functions into diverse classes and evaluates their roles in neural network training.

- [**Parametric Activation Functions for Neural Networks: A Tutorial Survey**](https://ieeexplore.ieee.org/abstract/document/10705284/)<br />
  _Authors_: Pusztaházi, L. S., Eigner, G., & Csiszár, O.<br />
  _Description_: A detailed tutorial on parametric activation functions, highlighting their adaptability and advantages over static counterparts.

---

### **4.3 Backpropagation and Gradient Descent**

Explore the mathematics and algorithms that drive neural network training.

- [**A Mathematical Theory of Communication**](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)<br />
  _Authors_: Claude Shannon<br />
  _Description_: This seminal work laid the foundation for information theory, which is crucial for neural networks.

- [**Learning Internal Representations by Error Propagation**](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf)<br />
  _Authors_: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams<br />
  _Description_: Introduced the backpropagation algorithm, a powerful method for training multi-layer perceptrons.

- [**On the Convergence Properties of the Back-Propagation Algorithm**](https://ieeexplore.ieee.org/document/118527)<br />
  _Authors_: Y. LeCun, L. D. Jackel, L. Bottou<br />
  _Description_: Investigates the convergence properties of the backpropagation algorithm, providing insights into its strengths and limitations.

- [**An overview of gradient descent optimization algorithms**](https://arxiv.org/pdf/1609.04747)<br />
  _Authors_: Sebastian Ruder<br />
  _Description_: Compares various gradient descent optimization algorithms, including standard gradient descent, Momentum, Adagrad, RMSprop, and Adam. It explores their mechanisms, advantages, and trade-offs, helping practitioners choose the best algorithm based on specific tasks. The paper also addresses challenges such as hyperparameter tuning and generalization in machine learning.

- [**Efficient Backprop**](https://www.researchgate.net/publication/2811922_Efficient_BackProp)<br />
  _Authors_: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner<br />
  _Description_: Explores techniques for improving the efficiency of backpropagation, which is crucial for training large neural networks.

- [**Asynchronous stochastic gradient descent with decoupled backpropagation and layer-wise updates**](https://ar5iv.org/html/2410.05985)<br />
  _Authors_: Cabrel Teguemne Fokam, Khaleelulla Khan Nazeer, Lukas König, David Kappel, Anand Subramoney<br />
  _Description_: Presents a novel asynchronous approach to stochastic gradient descent, which decouples backpropagation across layers to improve efficiency in deep networks.

- [**Generalizing Backpropagation for Gradient-Based Interpretability**](https://ar5iv.org/html/2307.03056)<br />
  _Authors_: Kevin Du, Lucas Torroba Hennigen, Niklas Stoehr, Alexander Warstadt, Ryan Cotterell<br />
  _Description_: Explores the concept of backpropagation and its generalization to understand gradient-based interpretability in machine learning models.

- [**Gradient Descent based Optimization Algorithms for Deep Learning Models Training**](https://arxiv.org/pdf/1903.03614v1)<br />
  _Authors_: Jiawei Zhang<br />
  _Description_: Explores gradient descent optimization techniques for training deep learning models, highlighting common methods like Momentum, Adagrad, Adam, and Gadam. It discusses how these algorithms improve training efficiency and performance, especially for complex models and high-dimensional data.

---

### **4.4 Optimization Techniques**

Learn about optimization methods that improve training efficiency and performance in deep learning.

- [**Optimization Techniques in Machine Learning and Deep Learning**](https://pijet.org/papers/volume-1%20issue-2/Final_Revised%20Paper_Pijet-14.pdf)<br />
  _Authors_: Ashutosh V. Patil, Gayatri Y. Bhangle<br />
  _Description_: Explores optimization techniques like gradient descent, its variants, and convergence properties.

- [**Optimization for deep learning: theory and algorithms**](https://ar5iv.org/html/1912.08957)<br />
  _Authors_: Ruoyu Sun<br />
  _Description_: Discusses optimization techniques for deep learning, with a focus on gradient descent and stochastic gradient descent (SGD).

- [**Optimization Methods in Deep Learning: A Comprehensive Overview**](https://www.researchgate.net/publication/368664550_Optimization_Methods_in_Deep_Learning_A_Comprehensive_Overview)<br />
  _Authors_: David Shulman<br />
  _Description_: Offers an extensive review of optimization techniques for deep learning, covering methods like gradient descent, SGD, and their variants. Provides insights into their mathematical foundations and practical applications.

- [**Advanced metaheuristic optimization techniques in applications of deep neural networks: a review**](https://www.researchgate.net/profile/Laith-Abualigah/publication/350959219_Advanced_metaheuristic_optimization_techniques_in_applications_of_deep_neural_networks_a_review/links/607ef83b881fa114b416729b/Advanced-metaheuristic-optimization-techniques-in-applications-of-deep-neural-networks-a-review.pdf)<br />
  _Authors_: Abd Elaziz, Mohamed; Dahou, Abdelghani; Abualigah, Laith; Yu, Liyang; Alshinwan, Mohammad; Khasawneh, Ahmad M; Lu, Songfeng<br />
  _Description_: Reviews advanced metaheuristic optimization techniques applied to deep neural networks, focusing on methods like genetic algorithms, particle swarm optimization, and simulated annealing to enhance DNN training efficiency.

---

## **5. Sequence Modeling**

This section explores models and techniques for handling sequential data, such as text, speech, or time-series, including RNNs, LSTMs, sequence-to-sequence models, attention mechanisms, and transformers.

---

### **5.1 RNNs and LSTMs**

Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are widely used for processing sequential data. Below are key papers on their development and applications:

- [**Recurrent neural network and LSTM models for lexical utterance classification**](https://www.isca-archive.org/interspeech_2015/ravuri15_interspeech.pdf)<br />
  _Authors_: SV Ravuri, A Stolcke<br />
  _Description_: This paper explores the application of RNN and LSTM models for lexical utterance classification, highlighting the effectiveness of LSTMs for long utterances and RNNs for shorter ones.

- [**Introduction to sequence learning models: RNN, LSTM, GRU**](https://www.researchgate.net/publication/350950396_Introduction_to_Sequence_Learning_Models_RNN_LSTM_GRU)<br />
  _Authors_: S Zargar<br />
  _Description_: An introduction to sequence learning models including RNN, LSTM, and GRU, focusing on their architectures and applications in sequence-based tasks.

- [**Performance evaluation of deep neural networks applied to speech recognition: RNN, LSTM, and GRU**](https://sciendo.com/pdf/10.2478/jaiscr-2019-0006)<br />
  _Authors_: A Shewalkar, D Nyavanandi, SA Ludwig<br />
  _Description_: This paper evaluates the performance of RNNs, LSTMs, and GRUs in speech recognition tasks, emphasizing LSTM's superior word error rate.

- [**TTS synthesis with bidirectional LSTM-based recurrent neural networks**](https://www.isca-archive.org/interspeech_2014/fan14_interspeech.pdf)<br />
  _Authors_: Y Fan, Y Qian, FL Xie, FK Soong<br />
  _Description_: A study on text-to-speech synthesis using bidirectional LSTM networks, demonstrating improved modeling of sequential data.

- [**Learning precise timing with LSTM recurrent networks**](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf)<br />
  _Authors_: FA Gers, NN Schraudolph, J Schmidhuber<br />
  _Description_: This paper introduces LSTM networks with peepholes and forget gates, showcasing their ability to handle precise timing in sequential data.

- [**A critical review of RNN and LSTM variants in hydrological time series predictions**](https://www.sciencedirect.com/science/article/pii/S2215016124003972)<br />
  _Authors_: M Waqas, UW Humphries<br />
  _Description_: A review of RNN and LSTM models applied to hydrological time series data, analyzing their strengths and limitations.

- [**RNN-LSTM: From applications to modeling techniques and beyond—Systematic review**](https://www.sciencedirect.com/science/article/pii/S1319157824001575)<br />
  _Authors_: SM Al-Selwi, MF Hassan, SJ Abdulkadir<br />
  _Description_: A systematic review of RNN-LSTM applications and modeling techniques across various domains.

- [**Understanding LSTM--A tutorial into long short-term memory recurrent neural networks**](https://arxiv.org/abs/1909.09586)<br />
  _Authors_: RC Staudemeyer, ER Morris<br />
  _Description_: A tutorial offering a detailed explanation of LSTM networks and their role in addressing long-term dependency challenges in RNNs.

- [**A review of recurrent neural networks: LSTM cells and network architectures**](https://direct.mit.edu/neco/article-abstract/31/7/1235/8500)<br />
  _Authors_: Y Yu, X Si, C Hu, J Zhang<br />
  _Description_: This review categorizes various LSTM architectures and their applications, highlighting improvements over standard RNNs.

- [**Learning to diagnose with LSTM recurrent neural networks**](https://arxiv.org/abs/1511.03677)<br />
  _Authors_: ZC Lipton<br />
  _Description_: This paper demonstrates the use of LSTM networks for medical diagnosis tasks, showing their capability to process sequential patient data effectively.

---

### **5.2 Sequence Models**

Sequence models, such as sequence-to-sequence (seq2seq) architectures, handle input-output pairs with sequential relationships. Below are key papers on this topic:

- [**An Analysis of 'Attention' in Sequence-to-Sequence Models**](https://www.isca-archive.org/interspeech_2017/prabhavalkar17b_interspeech.pdf)<br />
  _Authors_: R Prabhavalkar, TN Sainath, B Li, K Rao, N Jaitly<br />
  _Description_: This paper examines the role of attention mechanisms in sequence-to-sequence models, focusing on their impact on tasks like speech recognition and translation.

- [**Sequence Modeling with CTC**](https://distill.pub/2017/ctc/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=DataScience%20Digest)<br />
  _Authors_: A Hannun<br />
  _Description_: This work introduces connectionist temporal classification (CTC) for sequence modeling, illustrating its use in aligning sequences like audio-to-text without explicit alignments.

- [**Neural Machine Translation and Sequence-to-Sequence Models: A Tutorial**](https://arxiv.org/abs/1703.01619)<br />
  _Authors_: G Neubig<br />
  _Description_: A comprehensive tutorial covering sequence-to-sequence models in machine translation, including encoder-decoder structures and attention mechanisms.

- [**Deep Reinforcement Learning for Sequence-to-Sequence Models**](https://ieeexplore.ieee.org/abstract/document/8801910/)<br />
  _Authors_: Y Keneshloo, T Shi, N Ramakrishnan<br />
  _Description_: The paper explores the integration of reinforcement learning techniques with sequence-to-sequence models for improved performance.

- [**Seq2sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples**](https://aaai.org/ojs/index.php/AAAI/article/view/5767)<br />
  _Authors_: M Cheng, J Yi, PY Chen, H Zhang, CJ Hsieh<br />
  _Description_: This work analyzes the robustness of sequence-to-sequence models under adversarial attacks, proposing frameworks to evaluate their stability.

- [**A Causal Framework for Explaining the Predictions of Black-Box Sequence-to-Sequence Models**](https://arxiv.org/abs/1707.01943)<br />
  _Authors_: D Alvarez-Melis, TS Jaakkola<br />
  _Description_: The paper introduces a causal framework to understand and explain decisions made by black-box sequence-to-sequence models.

- [**Lingvo: A Modular and Scalable Framework for Sequence-to-Sequence Modeling**](https://arxiv.org/abs/1902.08295)<br />
  _Authors_: J Shen, P Nguyen, Y Wu, Z Chen, MX Chen<br />
  _Description_: Lingvo, an open-source framework by Google, enables scalable training of sequence-to-sequence models for tasks like speech recognition and translation.

---

### **5.3 Attention Mechanism**

Attention mechanisms enable models to focus on the most relevant parts of the input when making predictions. This subsection includes research on various attention techniques:

- [**Gaussian Prediction Based Attention for Online End-to-End Speech Recognition**](https://www.isca-archive.org/interspeech_2017/hou17b_interspeech.pdf)<br />
  _Authors_: J Hou, S Zhang, LR Dai<br />
  _Description_: This paper introduces a Gaussian prediction-based attention mechanism to improve online end-to-end speech recognition by refining sequence alignment.

- [**Pose-conditioned Spatio-temporal Attention for Human Action Recognition**](https://arxiv.org/pdf/1703.10106)<br />
  _Authors_: F Baradel, C Wolf, J Mille<br />
  _Description_: Proposes a spatio-temporal attention mechanism conditioned on pose features for effective human action recognition from RGB video sequences.

- [**Recurrent Attention Network on Memory for Aspect Sentiment Analysis**](https://aclanthology.org/D17-1047/)<br />
  _Authors_: P Chen, Z Sun, L Bing, W Yang<br />
  _Description_: This paper explores a recurrent attention network for aspect-level sentiment analysis by leveraging multiple attention mechanisms to focus on sentiment features.

- [**SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning**](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)<br />
  _Authors_: L Chen, H Zhang, J Xiao, L Nie<br />
  _Description_: A new attention mechanism that combines spatial and channel-wise attention is presented for improved image captioning performance.

- [**Gated Self-Matching Networks for Reading Comprehension and Question Answering**](https://aclanthology.org/P17-1018/)<br />
  _Authors_: W Wang, N Yang, F Wei, B Chang<br />
  _Description_: This paper introduces gated self-matching attention for question answering, leveraging passage and question alignment to refine representations.

- [**Residual Attention Network for Image Classification**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf)<br />
  _Authors_: F Wang, M Jiang, C Qian, S Yang<br />
  _Description_: Residual attention networks enhance image classification by incorporating a novel attention mechanism into a deep residual network.

- [**Paying More Attention to Attention: Improving Performance of Convolutional Neural Networks via Attention Transfer**](https://arxiv.org/pdf/1612.03928)<br />
  _Authors_: S Zagoruyko, N Komodakis<br />
  _Description_: This paper improves convolutional neural network performance by utilizing attention transfer between teacher and student models during training.

- [**Topic Aware Neural Response Generation**](https://ojs.aaai.org/index.php/AAAI/article/view/10981)<br />
  _Authors_: C Xing, W Wu, Y Wu, J Liu<br />
  _Description_: This study develops a topic-aware attention mechanism for generating conversational responses, effectively aligning dialogue content with contextual topics.

---

### **5.4 Transformers**

Transformers are state-of-the-art architectures in sequence modeling, built around the self-attention mechanism. Below are significant papers that outline their theory and applications:

- [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)<br />
  _Authors_: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin<br />
  _Description_: The foundational paper introducing the transformer architecture. It details self-attention, encoder-decoder structure, and positional encodings, which are pivotal in sequence modeling tasks.

- [**Understanding How Positional Encodings Work in Transformer Models**](https://aclanthology.org/2024.lrec-main.1478/)<br />
  _Authors_: T Miyazaki, H Mino, H Kaneko<br />
  _Description_: Examines the functionality of positional encodings in self-attention and cross-attention blocks of transformer architectures, exploring their integration in encoder-decoder models.

- [**Universal Transformers**](https://arxiv.org/abs/1807.03819)<br />
  _Authors_: M Dehghani, S Gouws, O Vinyals, J Uszkoreit<br />
  _Description_: Introduces a universal transformer that extends the standard model by incorporating recurrence in the self-attention mechanism, enhancing its theoretical depth and reasoning capabilities.

- [**Position Information in Transformers: An Overview**](https://direct.mit.edu/coli/article-pdf/48/3/733/2040503/coli_a_00445.pdf)<br />
  _Authors_: P Dufter, M Schmitt, H Schütze<br />
  _Description_: Systematically reviews positional encoding techniques in transformers, analyzing over 30 models to understand their role in encoding positional information for attention mechanisms.

- [**Transformer Working Memory Enables Regular Language Reasoning and Natural Language Length Extrapolation**](https://arxiv.org/abs/2305.03796)<br />
  _Authors_: TC Chi, TH Fan, AI Rudnicky, PJ Ramadge<br />
  _Description_: Explores how transformer working memory interacts with self-attention to enable reasoning in regular languages and length extrapolation in NLP tasks.

- [**Understanding the Failure of Batch Normalization for Transformers in NLP**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f4f2f2b3c67da711df6df557fc870c4a-Abstract-Conference.html)<br />
  _Authors_: J Wang, J Wu, L Huang<br />
  _Description_: Investigates the challenges batch normalization introduces to self-attention and proposes alternatives for stabilizing transformer training in NLP tasks.

- [**Activating Self-Attention for Multi-Scene Absolute Pose Regression**](https://arxiv.org/abs/2411.01443)<br />
  _Authors_: M Lee, J Kim, JP Heo<br />
  _Description_: Details the functionality of self-attention and positional encoding in transformer encoders and cross-attention modules, applied to multi-scene regression tasks.

- [**Aiatrack: Attention in Attention for Transformer Visual Tracking**](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_9)<br />
  _Authors_: S Gao, C Zhou, C Ma, X Wang, J Yuan<br />
  _Description_: Explores self-attention and cross-attention mechanisms within the encoder-decoder structure of transformers, focusing on applications in tracking tasks.

- [**Why Transformers Are Obviously Good Models of Language**](https://arxiv.org/abs/2408.03855)<br />
  _Authors_: F Hill<br />
  _Description_: Discusses theoretical justifications for transformers' success in NLP, emphasizing the role of self-attention and cross-attention in language modeling.

- [**Learning Deep Learning: Theory and Practice of Neural Networks, Transformers, and NLP**](https://books.google.com/books?hl=en&lr=&id=wNnPEAAAQBAJ)<br />
  _Authors_: M Ekman<br />
  _Description_: Provides a comprehensive overview of transformers' components, including detailed discussions on self-attention, cross-attention, and encoder-decoder interactions in NLP.

---

## **6. Word Representations**

Word representations are the foundation of many natural language processing tasks. This section is divided into three key areas: **Static Word Embeddings**, **Contextualized Embeddings**, and **Subword-Based Representations**, covering both classical and cutting-edge methods for representing words in vector spaces.

---

### **6.1 Static Word Embeddings**

Static word embeddings, such as Word2Vec, GloVe, and FastText, represent each word with a fixed vector. Below are notable papers discussing their applications and limitations:

- [**Evaluating the Effectiveness of Static Word Embeddings on the Classification of IT Support Tickets**](https://www.researchgate.net/profile/Yasmen-Wahba/publication/345734017_Evaluating_the_Effectiveness_of_Static_Word_Embeddings_on_the_Classification_of_IT_Support_Tickets/links/5fac1b52a6fdcc331b95481c/Evaluating-the-Effectiveness-of-Static-Word-Embeddings-on-the-Classification-of-IT-Support-Tickets.pdf)<br />
  _Authors_: Y. Wahba, N.H. Madhavji<br />
  _Description_: This paper evaluates the performance of static word embeddings in IT ticket classification, focusing on their semantic capturing capabilities and limitations in dynamic contexts.

- [**Static Detection of Malicious PowerShell Based on Word Embeddings**](https://www.sciencedirect.com/science/article/pii/S2542660521000482)<br />
  _Authors_: M. Mimura, Y. Tajiri<br />
  _Description_: Proposes a method for detecting malicious PowerShell scripts using static word embeddings, demonstrating their application in cybersecurity.

- [**Examining the Effect of Whitening on Static and Contextualized Word Embeddings**](https://www.sciencedirect.com/science/article/pii/S0306457323000092)<br />
  _Authors_: S. Sasaki, B. Heinzerling, J. Suzuki, K. Inui<br />
  _Description_: Analyzes how whitening techniques affect the quality and utility of static word embeddings compared to contextual embeddings.

- [**Obtaining Better Static Word Embeddings Using Contextual Embedding Models**](https://arxiv.org/abs/2106.04302)<br />
  _Authors_: P. Gupta, M. Jaggi<br />
  _Description_: Introduces a method to improve static word embeddings by distilling knowledge from contextual embedding models like BERT.

- [**A Survey on Training and Evaluation of Word Embeddings**](https://link.springer.com/article/10.1007/s41060-021-00242-8)<br />
  _Authors_: F. Torregrossa, R. Allesiardo, V. Claveau, N. Kooli<br />
  _Description_: Provides a comprehensive overview of the training, evaluation, and application of static word embeddings across various NLP tasks.

- [**Dynamic Word Embeddings for Evolving Semantic Discovery**](https://dl.acm.org/doi/abs/10.1145/3159652.3159703)<br />
  _Authors_: Z. Yao, Y. Sun, N. Rao, H. Xiong<br />
  _Description_: Discusses the evolution from static to dynamic embeddings, highlighting the limitations of static methods in capturing semantic changes over time.

- [**A Comprehensive Analysis of Static Word Embeddings for Turkish**](https://www.sciencedirect.com/science/article/pii/S0957417424009898)<br />
  _Authors_: K. Sarıtaş, C.A. Öz, T. Güngör<br />
  _Description_: Analyzes static word embeddings for Turkish language processing, exploring their performance and limitations compared to contextual embeddings.

- [**On Measuring and Mitigating Bias in Static Word Embeddings**](https://ojs.aaai.org/index.php/AAAI/article/view/6267)<br />
  _Authors_: S. Dev, T. Li, J.M. Phillips, V. Srikumar<br />
  _Description_: Investigates biases in static word embeddings and proposes mitigation strategies to reduce stereotypical inferences in NLP applications.

- [**Learning Sense-Specific Static Embeddings Using Contextualized Word Embeddings as a Proxy**](https://arxiv.org/abs/2110.02204)<br />
  _Authors_: Y. Zhou, D. Bollegala<br />
  _Description_: Explores creating sense-specific static embeddings by leveraging contextual embeddings to overcome polysemy in static models.

- [**Static Embeddings as Efficient Knowledge Bases?**](https://arxiv.org/abs/2104.07094)<br />
  _Authors_: P. Dufter, N. Kassner, H. Schütze<br />
  _Description_: Evaluates whether static word embeddings can serve as efficient knowledge bases, especially in low-resource scenarios.

---

### **6.2 Contextualized Embeddings**

Contextualized word embeddings, such as those generated by BERT, GPT, or ELMo, vary depending on the context in which the word appears. These embeddings capture semantic and syntactic nuances, making them ideal for a wide range of NLP tasks.

- [**Combining contextualized embeddings and prior knowledge for clinical named entity recognition**](https://medinform.jmir.org/2019/4/e14850)<br />
  _Authors_: M Jiang, T Sanger, X Liu<br />
  _Description_: This study integrates contextualized embeddings like BERT with domain-specific knowledge for clinical named entity recognition, showcasing its enhanced performance in the medical domain.

- [**How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings**](https://arxiv.org/abs/1909.00512)<br />
  _Authors_: K Ethayarajh<br />
  _Description_: Explores the degree of contextualization in BERT, ELMo, and GPT-2 embeddings by analyzing their geometry and comparing their ability to model semantic and syntactic nuances.

- [**What do you learn from context? Probing for sentence structure in contextualized word representations**](https://arxiv.org/abs/1905.06316)<br />
  _Authors_: I Tenney, P Xia, B Chen, A Wang, A Poliak<br />
  _Description_: Investigates how contextualized embeddings encode sentence structure, demonstrating their potential in diverse NLP tasks, such as part-of-speech tagging and syntax analysis.

- [**Evaluating the underlying gender bias in contextualized word embeddings**](https://arxiv.org/abs/1904.08783)<br />
  _Authors_: C Basta, MR Costa-Jussà, N Casas<br />
  _Description_: Analyzes biases in contextualized embeddings like BERT and ELMo, revealing their implicit gender biases and proposing mitigation strategies.

- [**Med-BERT: Pretrained contextualized embeddings for electronic health records**](https://www.nature.com/articles/s41746-021-00455-y)<br />
  _Authors_: L Rasmy, Y Xiang, Z Xie, C Tao, D Zhi<br />
  _Description_: Introduces Med-BERT, a contextualized embedding model trained on large-scale health records for disease prediction, enhancing performance in medical NLP tasks.

- [**Contextualized embeddings based transformer encoder for sentence similarity modeling**](https://aclanthology.org/2020.lrec-1.676/)<br />
  _Authors_: MTR Laskar, X Huang, E Hoque<br />
  _Description_: Applies contextualized embeddings in a transformer-based encoder architecture for sentence similarity tasks, yielding state-of-the-art results.

- [**A survey on contextual embeddings**](https://arxiv.org/abs/2003.07278)<br />
  _Authors_: Q Liu, MJ Kusner, P Blunsom<br />
  _Description_: Provides an extensive survey on contextualized embeddings, discussing their evolution, underlying mechanisms, and applications in NLP tasks.

- [**Does BERT make any sense? Interpretable word sense disambiguation with contextualized embeddings**](https://arxiv.org/abs/1909.10430)<br />
  _Authors_: G Wiedemann, S Remus, A Chawla<br />
  _Description_: Investigates BERT's ability to disambiguate word senses, comparing it to other contextualized embeddings and revealing its superior performance in capturing polysemy.

- [**Interpreting pretrained contextualized representations via reductions to static embeddings**](https://aclanthology.org/2020.acl-main.431/)<br />
  _Authors_: R Bommasani, K Davis, C Cardie<br />
  _Description_: Analyzes pretrained contextualized embeddings like BERT by reducing them to static representations, providing insights into their semantic structure.

- [**BERTRAM: Improved word embeddings have a big impact on contextualized model performance**](https://arxiv.org/abs/1910.07181)<br />
  _Authors_: T Schick, H Schütze<br />
  _Description_: Proposes BERTRAM, a technique for enhancing word embeddings, and examines its impact on improving the performance of contextualized models.

---

### **6.3 Subword-Based Representations**

Subword-based representations break down words into smaller units, such as character n-grams or byte pair encodings (BPE). These methods are particularly useful for handling rare or unseen words, as well as morphologically rich languages.

- [**Studies on Subword-based Low-Resource Neural Machine Translation: Segmentation, Encoding, and Decoding**](https://repository.kulib.kyoto-u.ac.jp/dspace/bitstream/2433/288857/1/djohk00861.pdf)<br />
  _Authors_: S Haiyue<br />
  _Description_: Explores the role of subword segmentation and encoding in low-resource machine translation, focusing on efficient training strategies for neural models.

- [**Effective Subword Segmentation for Text Comprehension**](https://ieeexplore.ieee.org/abstract/document/8735719/)<br />
  _Authors_: Z Zhang, H Zhao, J Li, Z Li<br />
  _Description_: Examines how subword-based frameworks improve robustness across languages for text comprehension tasks in NLP.

- [**Subword Attention and Post-Processing for Rare and Unknown Contextualized Embeddings**](https://aclanthology.org/2024.findings-naacl.88/)<br />
  _Authors_: R Patel, C Domeniconi<br />
  _Description_: Proposes a novel subword attention mechanism to enhance rare and unknown token embeddings in contextualized representations.

- [**Learning to Generate Word Representations Using Subword Information**](https://aclanthology.org/C18-1216/)<br />
  _Authors_: Y Kim, KM Kim, JM Lee, SK Lee<br />
  _Description_: Introduces a framework for generating word representations by leveraging subword-level information to enhance downstream tasks.

- [**Entropy-Based Subword Mining with an Application to Word Embeddings**](https://aclanthology.org/W18-1202/)<br />
  _Authors_: A El-Kishky, FF Xu, A Zhang, S Macke<br />
  _Description_: Presents a method to mine subword units using entropy-based segmentation, improving embeddings for low-resource languages.

- [**Subword-Based Compact Reconstruction for Open-Vocabulary Neural Word Embeddings**](https://aclanthology.org/N19-1353/)<br />
  _Authors_: S Sasaki, J Suzuki, K Inui<br />
  _Description_: Proposes a reconstruction technique for subword-based embeddings, enabling efficient modeling of open-vocabulary tasks in NLP.

- [**Patterns Versus Characters in Subword-Aware Neural Language Modeling**](https://arxiv.org/pdf/1709.00541)<br />
  _Authors_: R Takhanov, Z Assylbekov<br />
  _Description_: Compares subword-level modeling techniques with character-based approaches, focusing on their effectiveness in language modeling.

- [**Lexically Grounded Subword Segmentation**](https://arxiv.org/abs/2406.13560)<br />
  _Authors_: J Libovický, J Helcl<br />
  _Description_: Proposes a lexically grounded subword segmentation method to optimize subword tokenization for diverse NLP applications.

- [**The Use of Subwords for Automatic Speech Recognition**](https://skemman.is/bitstream/1946/39412/1/Or%C3%B0fl%C3%ADsar-DavidErikMollberg.pdf)<br />
  _Authors_: DE Mollberg<br />
  _Description_: Applies subword-based approaches to automatic speech recognition, evaluating their performance in Icelandic language processing.

- [**Analysis of Word Dependency Relations and Subword Models in Abstractive Text Summarization**](https://www.cmpe.boun.edu.tr/~gungort/theses/Analysis%20of%20Word%20Dependency%20Relations%20and%20Subword%20Models%20in%20Abstractive%20Text%20Summarization.pdf)<br />
  _Authors_: AB Özkan, T Güngör<br />
  _Description_: Analyzes the impact of subword models on abstractive text summarization tasks, particularly in morphologically complex languages.

---

## **7. Evaluation**

Evaluation is a critical aspect of Natural Language Processing (NLP) to assess the effectiveness, robustness, and fairness of models. This section covers evaluation metrics, model validation techniques, and fairness metrics that ensure NLP models are measured accurately and ethically.

---

### **7.1 Evaluation Metrics (Accuracy, BLEU, ROUGE, etc.)**

Evaluation metrics like BLEU, ROUGE, and METEOR are widely used to measure the quality of NLP systems, especially for tasks like summarization, machine translation, and text generation.

- [**Comparing automatic and human evaluation of NLG systems**](https://aclanthology.org/E06-1040.pdf)<br />
  _Authors_: A Belz, E Reiter<br />
  _Description_: This paper explores the strengths and weaknesses of automatic evaluation metrics such as BLEU and ROUGE in natural language generation (NLG) systems compared to human judgments.

- [**Re-evaluating automatic summarization with BLEU and 192 shades of ROUGE**](https://aclanthology.org/D15-1013.pdf)<br />
  _Authors_: Y Graham<br />
  _Description_: Investigates the performance of BLEU and ROUGE metrics in evaluating summarization tasks, with a focus on improving their correlation with human evaluations.

- [**Beyond ROUGE: A comprehensive evaluation metric for abstractive summarization leveraging similarity, entailment, and acceptability**](https://www.researchgate.net/publication/381267843)<br />
  _Authors_: MKH Briman, B Yildiz<br />
  _Description_: Proposes a new evaluation framework for abstractive summarization by incorporating similarity, entailment, and acceptability metrics beyond traditional n-gram-based metrics like ROUGE.

- [**An investigation into the validity of some metrics for automatically evaluating natural language generation systems**](https://direct.mit.edu/coli/article-pdf/35/4/529/1798673/coli.2009.35.4.35405.pdf)<br />
  _Authors_: E Reiter, A Belz<br />
  _Description_: Critically evaluates several metrics like BLEU and ROUGE, revealing their limitations as predictors of human judgment in NLG systems.

- [**A survey of evaluation metrics used for NLG systems**](https://arxiv.org/pdf/2008.12009)<br />
  _Authors_: AB Sai, AK Mohankumar, MM Khapra<br />
  _Description_: A comprehensive survey that compares commonly used evaluation metrics such as BLEU, ROUGE, and METEOR, providing insights into their use cases and limitations in natural language generation.

- [**Adaptations of ROUGE and BLEU to better evaluate machine reading comprehension tasks**](https://arxiv.org/abs/1806.03578)<br />
  _Authors_: A Yang, K Liu, J Liu, Y Lyu, S Li<br />
  _Description_: Proposes modifications to traditional ROUGE and BLEU metrics to better assess performance in machine reading comprehension tasks.

- [**A critical analysis of metrics used for measuring progress in artificial intelligence**](https://arxiv.org/pdf/2008.02577)<br />
  _Authors_: K Blagec, G Dorffner, M Moradi, M Samwald<br />
  _Description_: Analyzes the metrics used to measure progress in NLP, with a focus on BLEU, ROUGE, and other widely used evaluation methods.

- [**Evaluation of NLP systems**](http://www.umiacs.umd.edu/~jbg/teaching/CMSC_773_2012/reading/evaluation.pdf)<br />
  _Authors_: P Resnik, J Lin<br />
  _Description_: Discusses the theoretical and practical aspects of evaluating NLP systems using metrics like BLEU, ROUGE, precision, and recall.

- [**Comparison of evaluation metrics for short story generation**](https://ieeexplore.ieee.org/abstract/document/10329351/)<br />
  _Authors_: P Netisopakul, U Taoto<br />
  _Description_: Compares BLEU, ROUGE-L, and BERTScore as metrics for short story generation, providing insights into their effectiveness and limitations.

- [**Revisiting automatic evaluation of extractive summarization tasks: Can we do better than ROUGE?**](https://aclanthology.org/2022.findings-acl.122/)<br />
  _Authors_: M Akter, N Bansal, SK Karmaker<br />
  _Description_: Analyzes the limitations of ROUGE in extractive summarization tasks and explores alternative metrics that better correlate with human judgments.

---

### **7.2 Model Validation and Cross-validation in NLP**

Model validation ensures that NLP systems perform reliably across various datasets and settings. Techniques like cross-validation are crucial for optimizing models and preventing overfitting.

- [**Improving the classification accuracy using recursive feature elimination with cross-validation**](https://www.researchgate.net/publication/344181117_Improving_the_Classification_Accuracy_using_Recursive_Feature_Elimination_with_Cross-Validation)<br />
  _Authors_: P. Misra, A.S. Yadav<br />
  _Description_: Discusses the effectiveness of recursive feature elimination with cross-validation for optimizing feature selection and classification accuracy in NLP models.

- [**Natural language processing and machine learning methods to characterize unstructured patient-reported outcomes: validation study**](https://www.jmir.org/2021/11/e26777/)<br />
  _Authors_: Z. Lu, J.A. Sim, J.X. Wang, C.B. Forrest, K.R. Krull<br />
  _Description_: Applies 5-folder nested cross-validation to validate NLP models in analyzing patient-reported outcomes, comparing their predictive performance.

- [**On the need of cross-validation for discourse relation classification**](https://aclanthology.org/E17-2024/)<br />
  _Authors_: W. Shi, V. Demberg<br />
  _Description_: Explores the necessity of cross-validation in discourse relation classification, demonstrating its role in stabilizing performance in small evaluation datasets.

- [**Resumate: A prototype to enhance recruitment process with NLP-based resume parsing**](https://ieeexplore.ieee.org/abstract/document/10166169/)<br />
  _Authors_: S. Mishra<br />
  _Description_: Presents an NLP-based recruitment tool using k-fold cross-validation for robust evaluation of parsing models, ensuring improved generalization.

- [**Cross-validation visualized: a narrative guide to advanced methods**](https://www.mdpi.com/2504-4990/6/2/65)<br />
  _Authors_: J. Allgaier, R. Pryss<br />
  _Description_: Provides a comprehensive guide to advanced cross-validation techniques, focusing on time-split methods for NLP applications.

- [**Is my stance the same as your stance? A cross-validation study of stance detection datasets**](https://www.sciencedirect.com/science/article/pii/S0306457322001728)<br />
  _Authors_: L.H.X. Ng, K.M. Carley<br />
  _Description_: Analyzes cross-validation techniques for stance detection in NLP, exploring dataset-specific challenges and their impact on model performance.

- [**Using JK fold cross-validation to reduce variance when tuning NLP models**](https://arxiv.org/abs/1806.07139)<br />
  _Authors_: H.B. Moss, D.S. Leslie, P. Rayson<br />
  _Description_: Proposes JK-fold cross-validation as a method to reduce variance and improve robustness during hyperparameter tuning for NLP models.

- [**Validation of prediction models for critical care outcomes using natural language processing of electronic health record data**](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2719128)<br />
  _Authors_: B.J. Marafino, M. Park, J.M. Davies, R. Thombley<br />
  _Description_: Evaluates prediction models using nested cross-validation to minimize bias, applying NLP to extract features from clinical text.

- [**Development and validation of machine models using natural language processing to classify substances involved in overdose deaths**](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2794977)<br />
  _Authors_: D. Goodman-Meza, C.L. Shover, J.A. Medina<br />
  _Description_: Utilizes 10-fold cross-validation to validate NLP models that classify substances mentioned in overdose death reports.

- [**PhageAI-bacteriophage life cycle recognition with machine learning and natural language processing**](https://www.biorxiv.org/content/10.1101/2020.07.11.198606.abstract)<br />
  _Authors_: P. Tynecki, A. Guziński, J. Kazimierczak, M. Jadczuk<br />
  _Description_: Integrates NLP and machine learning with stratified shuffle and 10-fold cross-validation to predict bacteriophage life cycles.

---

### **7.3 Bias and Fairness Metrics**

Bias and fairness metrics evaluate how equitably NLP models perform across different groups and ensure that systems do not perpetuate or amplify societal biases.

- [**Measuring fairness with biased rulers: A comparative study on bias metrics for pre-trained language models**](https://lirias.kuleuven.be/retrieve/667403)<br />
  _Authors_: P. Delobelle, E.K. Tokpo, T. Calders<br />
  _Description_: This paper examines various bias metrics applied to pre-trained NLP models, highlighting their strengths, limitations, and experimental evaluations.

- [**Bipol: A novel multi-axes bias evaluation metric with explainability for NLP**](https://www.sciencedirect.com/science/article/pii/S2949719123000274)<br />
  _Authors_: L. Alkhaled, T. Adewumi, S.S. Sabry<br />
  _Description_: Introduces a novel metric to evaluate multiple dimensions of bias in NLP models while incorporating explainability for better transparency.

- [**Bias and fairness in large language models: A survey**](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00524/121961)<br />
  _Authors_: I.O. Gallegos, R.A. Rossi, J. Barrow, M.M. Tanjim<br />
  _Description_: Provides an extensive survey on fairness and bias in large language models, with emphasis on definitions, metrics, and their applications.

- [**Quantifying social biases in NLP: A generalization and empirical comparison of extrinsic fairness metrics**](https://direct.mit.edu/tacl/article-abstract/doi/10.1162/tacl_a_00425/108201)<br />
  _Authors_: P. Czarnowska, Y. Vyas, K. Shah<br />
  _Description_: Examines social bias metrics in NLP, unifying various fairness metrics under a generalized framework for better empirical understanding.

- [**On Measurements of Bias and Fairness in NLP**](http://research.google/pubs/on-measurements-of-bias-and-fairness-in-nlp/)<br />
  _Authors_: S. Dev, E. Sheng, J. Zhao<br />
  _Description_: A survey discussing bias measures in NLP, covering metrics, datasets, and societal implications of biases in language models.

- [**Advancing Fairness in Natural Language Processing: From Traditional Methods to Explainability**](https://arxiv.org/abs/2410.12511)<br />
  _Authors_: F. Jourdan<br />
  _Description_: Explores how explainability methods can address biases in NLP systems while assessing the effectiveness of standard fairness metrics.

- [**A survey on bias and fairness in natural language processing**](https://arxiv.org/abs/2204.09591)<br />
  _Authors_: R. Bansal<br />
  _Description_: Discusses sources of bias in NLP models and highlights fairness metrics and mitigation strategies tailored for NLP tasks.

- [**Bias Exposed: The BiaXposer Framework for NLP Fairness**](https://link.springer.com/chapter/10.1007/978-981-96-0805-8_22)<br />
  _Authors_: Y. Gaci, B. Benatallah, F. Casati<br />
  _Description_: Proposes a new framework for detecting and quantifying biases in NLP, focusing on disparities in task-specific model performance.

- [**Bold: Dataset and metrics for measuring biases in open-ended language generation**](https://dl.acm.org/doi/abs/10.1145/3442188.3445924)<br />
  _Authors_: J. Dhamala, T. Sun, V. Kumar, S. Krishna<br />
  _Description_: Introduces a dataset and metrics for analyzing biases in language generation models, focusing on their societal implications.

- [**Should fairness be a metric or a model? A model-based framework for assessing bias in machine learning pipelines**](https://dl.acm.org/doi/abs/10.1145/3641276)<br />
  _Authors_: J.P. Lalor, A. Abbasi, K. Oketch<br />
  _Description_: Proposes a model-based framework for bias assessment, comparing its effectiveness to traditional fairness metrics in NLP pipelines.

---

## **8. Tasks**

This section explores major NLP tasks, from foundational challenges like text classification and named entity recognition to advanced applications such as machine translation and question answering. Each task highlights methodologies, benchmarks, and state-of-the-art approaches that drive innovation in understanding, generating, and transforming human language computationally.

### **8.1 Text Generation**

The automated creation of human-like text, such as stories, dialogue, or code. Modern models generate context-aware content for chatbots, creative writing, or code completion, balancing coherence and creativity while minimizing repetition or factual errors.

- [**Generation - A New Frontier of Natural Language Processing?**](https://aclanthology.org/T87-1041.pdf)<br />
  _Authors_: A. Joshi<br />
  _Description_: Discusses the theoretical underpinnings of text generation in NLP, exploring its significance as a foundational component of linguistic processing.

- [**Automated Title Generation in English Language Using NLP**](https://www.researchgate.net/publication/312452399)<br />
  _Authors_: N. Sethi, P. Agrawal, V. Madaan, S.K. Singh<br />
  _Description_: Presents a methodological framework for generating concise and relevant titles from English text using NLP techniques.

- [**Applied Text Generation**](https://aclanthology.org/A92-1006.pdf)<br />
  _Authors_: O. Rambow, T. Korelsky<br />
  _Description_: Introduces a system for applying text generation to practical tasks, offering insights into its flexibility and adaptability across applications.

- [**The Survey: Text Generation Models in Deep Learning**](https://www.sciencedirect.com/science/article/pii/S1319157820303360)<br />
  _Authors_: T. Iqbal, S. Qureshi<br />
  _Description_: Provides an in-depth analysis of text generation models, discussing deep learning-based methods and their theoretical advancements.

- [**Controlled Text Generation with Adversarial Learning**](https://www.politesi.polimi.it/bitstream/10589/152269/3/2019_12_Betti.pdf)<br />
  _Authors_: F. Betti<br />
  _Description_: Explores conditional and controlled text generation, leveraging adversarial learning to refine outputs for specific contexts.

- [**Neural Text Generation: Past, Present, and Beyond**](https://arxiv.org/pdf/1803.07133)<br />
  _Authors_: S. Lu, Y. Zhu, W. Zhang, J. Wang, Y. Yu<br />
  _Description_: Surveys neural text generation, highlighting historical advancements, current methodologies, and future challenges.

- [**A Theoretical Analysis of the Repetition Problem in Text Generation**](https://ojs.aaai.org/index.php/AAAI/article/view/17520)<br />
  _Authors_: Z. Fu, W. Lam, A.M.C. So, B. Shi<br />
  _Description_: Presents a theoretical framework for addressing repetition in generated text, a common issue in neural language models.

- [**Natural Language Generation**](https://library.navoiy-uni.uz/files/clark%20e.%20handbook%20nlp.%202010..pdf#page=600)<br />
  _Authors_: E. Reiter<br />
  _Description_: Explores the fundamentals of natural language generation, detailing its applications and challenges in connecting linguistic theory with practical systems.

- [**Evaluation of Text Generation: A Survey**](https://arxiv.org/pdf/2006.14799)<br />
  _Authors_: A. Celikyilmaz, E. Clark, J. Gao<br />
  _Description_: Analyzes evaluation metrics for text generation, providing theoretical insights into how generated text quality is assessed in NLP.

- [**Pre-trained Language Models for Text Generation: A Survey**](https://arxiv.org/pdf/2201.05273)<br />
  _Authors_: J. Li, T. Tang, W.X. Zhao, J.Y. Nie, J.R. Wen<br />
  _Description_: Examines pre-trained language models for text generation, focusing on their underlying mechanisms and theoretical implications.

### **8.2 Text Classification**

Assigning labels (e.g., sentiment, topic) to text segments. Used to categorize emails, analyze opinions, or detect spam by training models to recognize patterns in unstructured data.

- [**Type of supervised text classification system for unstructured text comments using probability theory technique**](https://www.researchgate.net/publication/337010711)<br />
  _Authors_: S Sreedhar Kumar, ST Ahmed<br />
  _Description_: Introduces a probability-based text classifier designed for unstructured text, offering theoretical insights into text classification frameworks using probabilistic models.

- [**Graph-theoretic approaches to text classification**](https://pure.ulster.ac.uk/files/88215054/2020ShanavasNPhD.pdf)<br />
  _Authors_: N Shanavas<br />
  _Description_: Explores graph-theoretic models for text classification, integrating concepts from data mining, machine learning, and NLP to enhance classification accuracy.

- [**Text classification algorithms: A survey**](https://www.mdpi.com/2078-2489/10/4/150)<br />
  _Authors_: K Kowsari, K Jafari Meimandi, M Heidarysafa, S Mendu<br />
  _Description_: Provides a detailed survey of text classification algorithms, covering foundational theories, challenges, and the latest trends in NLP applications.

- [**Deep learning-based text classification: A comprehensive review**](https://arxiv.org/pdf/2004.03705)<br />
  _Authors_: S Minaee, N Kalchbrenner, E Cambria<br />
  _Description_: Reviews deep learning methods for text classification, highlighting theoretical advancements and the transition from traditional machine learning techniques.

- [**A discourse-aware neural network-based text model for document-level text classification**](https://journals.sagepub.com/doi/abs/10.1177/0165551517743644)<br />
  _Authors_: K Lee, S Han, SH Myaeng<br />
  _Description_: Examines the role of discourse structures in text classification using neural networks, leveraging rhetorical structure theory for document-level analysis.

- [**Semantic text classification: A survey of past and recent advances**](https://www.sciencedirect.com/science/article/pii/S0306457317305757)<br />
  _Authors_: B Altınel, MC Ganiz<br />
  _Description_: Discusses semantic-based text classification techniques, comparing traditional methods with semantic-aware models for improved context handling.

- [**An introduction to a new text classification and visualization for natural language processing using topological data analysis**](https://arxiv.org/abs/1906.01726)<br />
  _Authors_: N Elyasi, MH Moghadam<br />
  _Description_: Proposes a novel approach to text classification using topological data analysis, offering unique visualizations for text categorization.

- [**Comparing BERT against traditional machine learning text classification**](https://arxiv.org/pdf/2005.13012)<br />
  _Authors_: S González-Carvajal, EC Garrido-Merchán<br />
  _Description_: Evaluates BERT's effectiveness in text classification compared to traditional models, providing insights into its theoretical and practical implications.

- [**Theory-guided multiclass text classification in online academic discussions**](https://www.researchgate.net/publication/381940372)<br />
  _Authors_: E Eryilmaz, B Thoms, Z Ahmed<br />
  _Description_: Combines theoretical frameworks with practical applications to enhance multiclass text classification in academic discussions.

- [**Naive Bayes and text classification: Introduction and theory**](https://arxiv.org/abs/1410.5329)<br />
  _Authors_: S Raschka<br />
  _Description_: Provides a comprehensive overview of the Naive Bayes classifier, focusing on its theoretical underpinnings and applications in text categorization.

### **8.3 Named Entity Recognition (NER)**

Identifying and classifying entities (e.g., people, locations) in text. Critical for extracting structured information from documents, enabling applications like search optimization and knowledge graph construction.

- [**Named entity recognition using support vector machine: A language independent approach**](https://www.researchgate.net/publication/228566833)<br />
  _Authors_: A. Ekbal, S. Bandyopadhyay<br />
  _Description_: Explores a language-independent approach to NER using support vector machines, emphasizing the theoretical basis of statistical learning for NLP tasks.

- [**Named entity recognition by using maximum entropy**](https://www.researchgate.net/publication/275829974)<br />
  _Authors_: I. Ahmed, R. Sathyaraj<br />
  _Description_: Demonstrates the application of maximum entropy modeling for NER, providing insights into probabilistic approaches to text classification.

- [**Named entity recognition: Fallacies, challenges, and opportunities**](https://www.sciencedirect.com/science/article/pii/S0920548912001080)<br />
  _Authors_: M. Marrero, J. Urbano, S. Sánchez-Cuadrado<br />
  _Description_: Analyzes the evolution of NER techniques, addressing theoretical fallacies and practical challenges in developing robust models.

- [**A comprehensive study of named entity recognition in Chinese clinical text**](https://academic.oup.com/jamia/article/21/5/808)<br />
  _Authors_: B. Tang, M. Jiang<br />
  _Description_: Focuses on applying NER to Chinese clinical text using discriminative statistical algorithms, bridging probability theory and NLP practice.

- [**A survey on deep learning for named entity recognition**](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10193)<br />
  _Authors_: J. Li, A. Sun, J. Han, C. Li<br />
  _Description_: Explores the use of deep learning techniques for NER, including recurrent and transformer-based models, highlighting theoretical advancements.

- [**Biomedical named entity recognition: A survey of machine-learning tools**](https://www.intechopen.com/chapters/38735)<br />
  _Authors_: D. Campos, S. Matos, J.L. Oliveira<br />
  _Description_: Provides a detailed survey of machine-learning approaches to NER, with a focus on biomedical text and domain-specific challenges.

- [**Theory and applications for biomedical named entity recognition without labeled data**](https://dl.acm.org/doi/10.1145/3558100.3563855)<br />
  _Authors_: X. Wei, L. Salsabil, J. Wu<br />
  _Description_: Proposes a distant supervision framework for NER in biomedical sciences, emphasizing theoretical underpinnings of weakly supervised learning.

- [**Named entity recognition and classification: State-of-the-art**](https://dl.acm.org/doi/10.1145/3445965)<br />
  _Authors_: Z. Nasar, S.W. Jaffry, M.K. Malik<br />
  _Description_: Offers a state-of-the-art review of NER techniques, covering theoretical foundations and their integration with relation extraction.

- [**Named entity recognition in the open domain**](https://www.degruyter.com/document/doi/10.1075/cilt.260.29eva)<br />
  _Authors_: R.J. Evans<br />
  _Description_: Discusses a framework for open-domain NER, highlighting challenges in generalization and theoretical approaches to multi-domain adaptability.

- [**Named entity recognition and classification in historical documents: A survey**](https://dl.acm.org/doi/10.1145/3604931)<br />
  _Authors_: M. Ehrmann, A. Hamdi, E.L. Pontes, M. Romanello<br />
  _Description_: Reviews the use of NER in historical document processing, exploring theoretical and methodological advancements for multilingual corpora.

### **8.4 Question Answering**

Answering natural language questions by extracting or generating responses from a given context. Powers virtual assistants and tools requiring precise retrieval of facts or reasoning over multiple sources.

- [**A survey of text question answering techniques**](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a3f655bc81a262dbe11a8f6fba575a454869022e)<br />
  _Authors_: P. Gupta, V. Gupta<br />
  _Description_: Provides an overview of text-based question answering systems, discussing core theoretical techniques and their application in natural language processing.

- [**Question answering from structured knowledge sources**](https://www.sciencedirect.com/science/article/pii/S157086830500090X)<br />
  _Authors_: A. Frank, H.U. Krieger, F. Xu, H. Uszkoreit<br />
  _Description_: Focuses on utilizing structured knowledge bases for question answering, incorporating graph-theoretical and NLP approaches to enhance accuracy.

- [**An application of automated reasoning in natural language question answering**](https://www.academia.edu/download/69950966/An_application_of_automated_reasoning_in20210919-1799-puv41g.pdf)<br />
  _Authors_: U. Furbach, I. Glöckner, B. Pelzer<br />
  _Description_: Integrates automated reasoning and theorem proving with NLP to develop a robust framework for question answering systems.

- [**Natural language question answering: the view from here**](https://www.cambridge.org/core/journals/natural-language-engineering/article/natural-language-question-answering-the-view-from-here/95EA883AFC7EB2B8EC050D3920F39DE2)<br />
  _Authors_: L. Hirschman, R. Gaizauskas<br />
  _Description_: Examines theoretical and practical advancements in question answering, emphasizing its role as a testbed for broader NLP research.

- [**Qa dataset explosion: A taxonomy of NLP resources for question answering**](https://dl.acm.org/doi/abs/10.1145/3560260)<br />
  _Authors_: A. Rogers, M. Gardner, I. Augenstein<br />
  _Description_: Categorizes datasets for question answering tasks, highlighting the theoretical implications of resource creation in NLP.

- [**A hyperintensional theory of intelligent question answering in TIL**](https://www.researchgate.net/profile/Marie-Duzi/publication/350378659_A_Hyperintensional_Theory_of_Intelligent_Question_Answering_in_TIL/links/60663586458515614d2b6e75/A-Hyperintensional-Theory-of-Intelligent-Question-Answering-in-TIL.pdf)<br />
  _Authors_: M. Duží, M. Fait<br />
  _Description_: Presents a hyperintensional framework for intelligent question answering, integrating formal semantics and logical reasoning.

- [**MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies**](https://www.sciencedirect.com/science/article/pii/S0306457315000515)<br />
  _Authors_: A.B. Abacha, P. Zweigenbaum<br />
  _Description_: Develops a medical question-answering system, blending NLP methods with semantic web principles to address domain-specific challenges.

- [**Revisiting the evaluation of theory of mind through question answering**](https://aclanthology.org/D19-1598.pdf)<br />
  _Authors_: M. Le, Y.L. Boureau, M. Nickel<br />
  _Description_: Investigates question answering as a means of evaluating cognitive models, including the theory of mind, in computational settings.

- [**The process of question answering: A computer simulation of cognition**](https://www.taylorfrancis.com/books/mono/10.4324/9781003316817)<br />
  _Authors_: W.G. Lehnert<br />
  _Description_: Simulates cognitive processes underlying question answering, linking general NLP theories with domain-specific implementation.

- [**Practical natural language processing question answering using graphs**](https://search.proquest.com/openview/710a291671577e8382dc5b514ffc35ed/1?pq-origsite=gscholar&cbl=18750&diss=y)<br />
  _Authors_: G.E. Fuchs<br />
  _Description_: Explores graph-based approaches to question answering, emphasizing the integration of conceptual graphs with NLP techniques.

### **8.5 Fill Mask**

A pre-training task where models predict masked words in sentences. Helps learn contextual relationships between words, forming the basis for training robust language models like BERT.

- [**The Fill-Mask Association Test (FMAT): Measuring Propositions in Natural Language**](https://psychbruce.github.io/paper/Bao_Accepted_JPSP_FMAT_Manuscript.pdf)<br />
  _Authors_: L. Lin, B. Wang, X. Wang, A. Wiśniowski<br />
  _Description_: Introduces FMAT for evaluating the probabilities of words in fill-mask tasks, exploring implications for understanding propositions in NLP.

- [**A Feature-Based Approach to Multilingual Idiomaticity Detection**](https://researchportal.helsinki.fi/files/220190492/2022.semeval_1.14.pdf)<br />
  _Authors_: S. Itkonen, J. Tiedemann<br />
  _Description_: Presents multilingual fill-mask tasks for detecting idiomaticity in language models, using features extracted from HuggingFace transformers.

- [**HuggingFace's Impact on Medical Applications of Artificial Intelligence**](https://www.sciencedirect.com/science/article/pii/S2950363924000036)<br />
  _Authors_: M. Riva, T.L. Parigi, F. Ungaro, L. Massimino<br />
  _Description_: Explores the application of fill-mask models in medical text processing, leveraging HuggingFace tools for advanced NLP applications.

- [**We Understand Elliptical Sentences, and Language Models Should Too**](https://aclanthology.org/2023.acl-long.188/)<br />
  _Authors_: D. Testa, E. Chersoni, A. Lenci<br />
  _Description_: Examines ellipsis resolution in NLP through fill-mask tasks, analyzing thematic fit and sentence structures for better language understanding.

- [**Time Masking for Temporal Language Models**](https://dl.acm.org/doi/abs/10.1145/3488560.3498529)<br />
  _Authors_: G.D. Rosin, I. Guy, K. Radinsky<br />
  _Description_: Investigates time masking in temporal language models, extending fill-mask tasks to predict temporal elements in NLP datasets.

- [**PronounFlow: A Hybrid Approach for Calibrating Pronouns in Sentences**](https://arxiv.org/abs/2308.15235)<br />
  _Authors_: N. Isaak<br />
  _Description_: Focuses on fill-mask tasks for refining pronoun usage in NLP systems, introducing hybrid calibration techniques for improved consistency.

- [**Homonym Sense Disambiguation in the Georgian Language**](https://arxiv.org/abs/2405.00710)<br />
  _Authors_: D. Melikidze, A. Gamkrelidze<br />
  _Description_: Presents a fill-mask model for resolving homonym ambiguities in Georgian, with applications to multilingual NLP tasks.

- [**Detection and Replacement of Neologisms for Translation**](https://search.proquest.com/openview/1be717392643ea691fa02b3821920bd7)<br />
  _Authors_: J. Pyo<br />
  _Description_: Uses fill-mask tasks for detecting and replacing neologisms in translations, ensuring accuracy and fluency in multilingual text processing.

- [**Mastering Transformers: Practical Applications of Fill-Mask in NLP**](https://books.google.com/books?hl=en&lr=&id=A9Y6EAAAQBAJ)<br />
  _Authors_: S. Yıldırım, M. Asgari-Chenaghlu<br />
  _Description_: Comprehensive guide to transformer models, highlighting fill-mask tasks for practical NLP applications in multilingual and domain-specific contexts.

- [**Towards Trustworthy NLP: Robustness Enhancement via Perplexity Difference**](https://ebooks.iospress.nl/doi/10.3233/FAIA230347)<br />
  _Authors_: Z. Ge, H. Hu, T. Zhao<br />
  _Description_: Proposes robustness improvement for fill-mask tasks using perplexity difference measures, ensuring reliability in NLP applications.

### **8.6 Machine Translation**

Translating text between languages while preserving meaning. Advances in neural models enable fluent translations, addressing challenges like idiomatic expressions and low-resource language support.

- [**The History of Natural Language Processing and Machine Translation**](https://staffwww.dcs.shef.ac.uk/people/L.Moffatt/yw_pubs/History_of_NLP%26MT_2004_final.pdf)<br />
  _Authors_: Y. Wilks<br />
  _Description_: Provides a historical overview of machine translation as a critical component of NLP, emphasizing its theoretical and practical evolution.

- [**Theoretical Overview of Machine Translation**](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=aad01b2a642711ef0b4d7d89d8d50fc268a222ce)<br />
  _Authors_: M.A. Chéragui<br />
  _Description_: Explores the theoretical foundations of machine translation, covering rule-based, statistical, and neural approaches in depth.

- [**Machine Translation Based on Type Theory**](https://www.cse.chalmers.se/alumni/janna/NLP/NLPtermpaper.pdf)<br />
  _Authors_: J. Khegai<br />
  _Description_: Investigates type theory as a framework for improving machine translation models, focusing on abstract and concrete syntax separation.

- [**Machine Translation and Philosophy of Language**](https://aclanthology.org/1994.bcs-1.26.pdf)<br />
  _Authors_: A.K. Melby<br />
  _Description_: Examines the philosophical implications of machine translation, linking language philosophy to the development of NLP methodologies.

- [**A Statistical Approach to Machine Translation**](https://aclanthology.org/J90-2002.pdf)<br />
  _Authors_: P.F. Brown, J. Cocke, S.A. Della Pietra<br />
  _Description_: Presents a foundational study on statistical machine translation, introducing techniques that influenced modern NLP approaches.

- [**Progress in Machine Translation**](https://www.sciencedirect.com/science/article/pii/S2095809921002745)<br />
  _Authors_: H. Wang, H. Wu, Z. He, L. Huang, K.W. Church<br />
  _Description_: Covers advancements in machine translation, from rule-based to neural models, highlighting breakthroughs in NLP systems.

- [**A Survey on Document-Level Neural Machine Translation**](https://arxiv.org/pdf/1912.08494)<br />
  _Authors_: S. Maruf, F. Saleh, G. Haffari<br />
  _Description_: Focuses on document-level neural machine translation, addressing contextual dependencies and evaluation challenges.

- [**An Optimized Cognitive-Assisted Machine Translation Approach for NLP**](https://link.springer.com/article/10.1007/s00607-019-00741-4)<br />
  _Authors_: A. Alarifi, A. Alwadain<br />
  _Description_: Proposes a cognitive-assisted machine translation framework, integrating NLP theories with cognitive modeling.

- [**Multilingual Natural Language Processing Applications: From Theory to Practice**](https://books.google.com/books?id=-eHztm8JRxwC)<br />
  _Authors_: D. Bikel, I. Zitouni<br />
  _Description_: Explores multilingual NLP with a focus on machine translation, detailing its theoretical underpinnings and practical applications.

- [**Machine Translation: A Knowledge-Based Approach**](https://dl.acm.org/doi/abs/10.5555/562174)<br />
  _Authors_: S. Nirenburg, J. Carbonell, M. Tomita<br />
  _Description_: Advances a knowledge-based methodology for machine translation, emphasizing its integration with domain-specific NLP tasks.

---

## **9. Models**

This section provides an overview of popular NLP models, ranging from foundational architectures to state-of-the-art models used for tasks like language generation, translation, classification, and more. Each model includes a brief description of its purpose, capabilities, and advancements.

### **9.1 BERT**

**BERT (Bidirectional Encoder Representations from Transformers)** is a revolutionary transformer-based model developed by Google. Unlike traditional models, BERT uses bidirectional context, allowing it to capture dependencies from both left and right sides of a token. It is widely used for tasks like text classification, question answering, and named entity recognition.

- [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805)<br />
  _Authors_: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova<br />
  _Description_: This groundbreaking paper introduced BERT, a bi-directional transformer-based model for language representation. It leverages masked language modeling and next sentence prediction tasks for pre-training, setting a new benchmark in numerous NLP tasks.

- [**Conditional BERT Contextual Augmentation**](https://arxiv.org/pdf/1812.06705)<br />
  _Authors_: Wu, Lv, Zang, Han<br />
  _Description_: Explores fine-tuning BERT for conditional text generation, showcasing its adaptability across NLP applications.

- [**BERT: A Review of Applications in NLP**](https://arxiv.org/pdf/2103.11943)<br />
  _Authors_: Koroteev, MV<br />
  _Description_: A comprehensive review of BERT’s applications in natural language understanding and processing.

- [**BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**](https://arxiv.org/pdf/2002.02925)<br />
  _Authors_: Xu, Zhou, Ge, Wei<br />
  _Description_: Investigates methods to compress BERT for lightweight deployments without significant performance loss.

### **9.2 GPT-3 (GPT)**

**GPT-3 (Generative Pre-trained Transformer 3)**, developed by OpenAI, is a large language model known for its impressive ability to generate coherent, human-like text. GPT-3 is widely used for tasks like text completion, question answering, and creative content generation. It builds on the generative pre-training concept introduced in GPT-2.

- [**Language Models Are Few-Shot Learners**](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)<br />
  _Authors_: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei<br />
  _Description_: This seminal paper introduces GPT-3, a large-scale transformer-based language model. It demonstrates state-of-the-art performance on a variety of NLP tasks using few-shot, one-shot, and zero-shot learning paradigms.

- [**What Makes Good In-Context Examples for GPT-3?**](https://arxiv.org/pdf/2101.06804)<br />
  _Authors_: J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin<br />
  _Description_: Investigates the effectiveness of example selection in few-shot settings for GPT-3, offering theoretical insights and practical strategies for better performance.

- [**Who is GPT-3? An Exploration of Personality, Values, and Demographics**](https://arxiv.org/pdf/2209.14338)<br />
  _Authors_: M. Miotto, N. Rossberg, B. Kleinberg<br />
  _Description_: Explores the personality and ethical considerations of GPT-3 by analyzing its outputs and implicit biases.

- [**GPT-3: Implications and Challenges for Machine Text**](https://arxiv.org/pdf/2107.01294)<br />
  _Authors_: Y. Dou, M. Forbes, R. Koncel-Kedziorski<br />
  _Description_: Evaluates the text generated by GPT-3 for linguistic and stylistic coherence, and highlights challenges in distinguishing machine-generated text from human-written content.

### **9.3 GPT-2**

**GPT-2 (Generative Pre-trained Transformer 2)** is the predecessor to GPT-3, with fewer parameters but still a powerful model for text generation. GPT-2 demonstrated the potential of transformer-based models to generate coherent and contextually relevant text, sparking advancements in generative AI.

- [**Language Models Are Unsupervised Multitask Learners**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)<br />
  _Authors_: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever<br />
  _Description_: GPT-2 introduces the concept of a large-scale transformer model pre-trained on diverse data. Its primary innovation lies in achieving strong performance on various NLP tasks without task-specific fine-tuning.

- [**Exploring the potential of GPT-2 for generating fake reviews of research papers**](https://ebooks.iospress.nl/doi/10.3233/FAIA200717)<br />
  _Authors_: A. Bartoli, E. Medvet<br />
  _Description_: Analyzes GPT-2's capabilities in generating synthetic text for specific use cases, including academic contexts.

- [**Hello, it's GPT-2: Towards the use of pretrained language models for task-oriented dialogue systems**](https://arxiv.org/abs/1907.05774)<br />
  _Authors_: P. Budzianowski, I. Vulić<br />
  _Description_: Explores task-oriented applications of GPT-2, emphasizing its use in dialogue systems.

- [**Feature-based detection of automated language models: Tackling GPT-2, GPT-3, and Grover**](https://peerj.com/articles/cs-443/)<br />
  _Authors_: L. Fröhling, A. Zubiaga<br />
  _Description_: Investigates methods to detect machine-generated text, highlighting challenges posed by models like GPT-2.

### **9.4 RoBERTa**

**RoBERTa (Robustly Optimized BERT Pretraining Approach)** is an improved version of BERT developed by Facebook AI. It modifies the pretraining process with larger datasets, longer training times, and other optimizations, resulting in improved performance across many NLP tasks.

- [**RoBERTa: A Robustly Optimized BERT Pretraining Approach**](https://arxiv.org/abs/1907.11692)<br />
  _Authors_: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov<br />
  _Description_: This paper enhances the BERT model by optimizing pretraining strategies, such as dynamic masking, increased training data, and larger batch sizes. RoBERTa outperforms BERT on multiple benchmarks, showcasing the benefits of improved pretraining techniques.

- [**Sentiment Classification with Modified RoBERTa and RNNs**](https://link.springer.com/article/10.1007/s11042-023-16833-5)<br />
  _Authors_: R. Cheruku, K. Hussain, I. Kavati, A.M. Reddy<br />
  _Description_: Demonstrates the use of RoBERTa in combination with recurrent neural networks to improve sentiment analysis.

- [**Robust Multilingual NLU with RoBERTa**](https://arxiv.org/pdf/2101.06804)<br />
  _Authors_: A. Conneau, A. Lample<br />
  _Description_: Extends RoBERTa's capabilities to multilingual natural language understanding tasks, showing its flexibility across languages.

- [**Aspect-Based Sentiment Analysis Using RoBERTa**](https://arrow.tudublin.ie/scschcomdis/232/)<br />
  _Authors_: G.R. Narayanaswamy<br />
  _Description_: Explores how RoBERTa can enhance sentiment classification with a focus on aspect-based analysis.

### **9.5 T5**

**T5 (Text-to-Text Transfer Transformer)**, developed by Google, frames every NLP task as a text-to-text problem. This unified approach allows T5 to perform tasks like translation, summarization, and question answering with remarkable efficiency and flexibility.

- [**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](https://arxiv.org/abs/1910.10683)<br />
  _Authors_: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu<br />
  _Description_: T5 introduces a unified framework where all NLP tasks are cast as text-to-text problems. It showcases exceptional performance across tasks by leveraging extensive pretraining on a diverse corpus.

- [**Clinical-T5: Large Language Models Built Using MIMIC Clinical Text**](https://www.physionet.org/content/clinical-t5/1.0.0/)<br />
  _Authors_: E. Lehman, A. Johnson<br />
  _Description_: Adapts the T5 model to the medical domain using MIMIC data, highlighting its potential in domain-specific applications.

- [**Deep Learning-Based Question Generation Using T5 Transformer**](https://link.springer.com/chapter/10.1007/978-981-16-0401-0_18)<br />
  _Authors_: K. Grover, K. Kaur, K. Tiwari, Rupali, P. Kumar<br />
  _Description_: Explores the application of T5 in generating questions for educational and interactive NLP tasks.

- [**Ptt5: Pretraining and Validating the T5 Model on Brazilian Portuguese Data**](https://arxiv.org/abs/2008.09144)<br />
  _Authors_: D. Carmo, M. Piau, I. Campiotti, R. Nogueira<br />
  _Description_: Adapts T5 for Portuguese, demonstrating its flexibility for multilingual and culturally specific applications.

### **9.6 DistilBERT**

**DistilBERT** is a smaller, faster, and more lightweight version of BERT. Developed by Hugging Face, it uses knowledge distillation to retain most of BERT's accuracy while reducing its size and computational requirements, making it suitable for real-time applications.

- [**DistilBERT: A Distilled Version of BERT**](https://arxiv.org/abs/1910.01108)<br />
  _Authors_: Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf<br />
  _Description_: This paper introduces DistilBERT, a lightweight version of BERT that achieves 97% of BERT’s performance while being 40% smaller and 60% faster, using a knowledge distillation technique.

- [**Online News Sentiment Classification Using DistilBERT**](https://cdn.techscience.cn/ueditor/files/jqc/TSP_JQC-4-1/TSP_JQC_26658/TSP_JQC_26658.pdf)<br />
  _Authors_: S.K. Akpatsa, H. Lei, X. Li, V.H.K.S. Obeng<br />
  _Description_: Explores DistilBERT’s efficiency in classifying online news sentiment, achieving high accuracy with minimal computational cost.

- [**Deep Question Answering: A New Teacher For DistilBERT**](https://core.ac.uk/download/pdf/294761173.pdf)<br />
  _Authors_: F. Tamburini, P. Cimiano, S. Preite<br />
  _Description_: Investigates how DistilBERT performs in question-answering tasks, emphasizing its learning from a BERT-based teacher.

- [**A Study of DistilBERT-Based Answer Extraction Machine Reading Comprehension Algorithm**](https://dl.acm.org/doi/abs/10.1145/3672919.3672968)<br />
  _Authors_: B. Li<br />
  _Description_: Proposes a DistilBERT-based machine reading comprehension model for accurate and efficient answer extraction.

### **9.7 ALBERT**

**ALBERT (A Lite BERT)** is a smaller and more efficient variant of BERT. It reduces the number of parameters through techniques like factorized embedding parameterization and shared parameters across layers, achieving faster training and inference without significant performance loss.

- [**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**](https://arxiv.org/abs/1909.11942)<br />
  _Authors_: Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut<br />
  _Description_: This paper introduces ALBERT, a lightweight and efficient variant of BERT. ALBERT reduces model size significantly while maintaining state-of-the-art performance using parameter sharing and factorized embeddings.

- [**Performance and Scalability of ALBERT in Question Answering Tasks**](https://arxiv.org/pdf/2003.10879)<br />
  _Authors_: J. Liu, Z. Zhao, T. Chen<br />
  _Description_: Explores the use of ALBERT in question-answering tasks, highlighting its efficiency and scalability across diverse datasets.

- [**ALBERT for Biomedical Named Entity Recognition**](https://dl.acm.org/doi/abs/10.1145/3487606.3487643)<br />
  _Authors_: H. Wang, S. Wu, R. Zhang<br />
  _Description_: Adapts ALBERT to biomedical NLP tasks, demonstrating its effectiveness in named entity recognition for domain-specific datasets.

- [**Efficient Fine-tuning with ALBERT**](https://arxiv.org/abs/2010.02820)<br />
  _Authors_: Y. Chen, F. Zhang, S. Guo<br />
  _Description_: Proposes strategies for efficient fine-tuning of ALBERT, showcasing reduced computational costs and improved adaptability.

### **9.8 BART**

**BART (Bidirectional and Auto-Regressive Transformers)**, developed by Facebook AI, is a versatile transformer model designed for text generation tasks. It combines the strengths of both bidirectional models like BERT and auto-regressive models like GPT, making it effective for summarization, translation, and more.

- [**BART: Denoising Sequence-to-Sequence Pretraining for Natural Language Generation, Translation, and Comprehension**](https://arxiv.org/abs/1910.13461)<br />
  _Authors_: Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer<br />
  _Description_: This paper introduces BART, a sequence-to-sequence model pre-trained with a denoising autoencoder approach. BART achieves state-of-the-art results on various NLP tasks, including summarization and machine translation.

- [**Abstractive English Document Summarization Using BART Model with Chunk Method**](https://www.sciencedirect.com/science/article/pii/S1877050924031375)<br />
  _Authors_: D. Suhartono, P. Wilman, T. Atara<br />
  _Description_: Explores the use of the BART model for abstractive document summarization, introducing a chunk-based methodology for improved performance.

- [**Fine-Tuning BART for Abstractive Reviews Summarization**](https://link.springer.com/chapter/10.1007/978-981-19-7346-8_32)<br />
  _Authors_: H. Yadav, N. Patel, D. Jani<br />
  _Description_: Presents fine-tuning techniques for BART to enhance its performance on abstractive summarization tasks, using Amazon reviews as a dataset.

- [**Template-Based Named Entity Recognition Using BART**](https://arxiv.org/abs/2106.01760)<br />
  _Authors_: L. Cui, Y. Wu, S. Yang, Y. Zhang<br />
  _Description_: Introduces a template-based approach for named entity recognition, leveraging BART's generative capabilities.

- [**Error Analysis of Using BART for Multi-Document Summarization**](https://aclanthology.org/2021.nodalida-main.43/)<br />
  _Authors_: T. Johner, A. Jana, C. Biemann<br />
  _Description_: Analyzes the performance of BART for multi-document summarization tasks, focusing on its application to English and German text.

### **9.9 ELECTRA**

**ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)** is an alternative to masked language modeling. Instead of masking tokens, it trains a model to detect replaced tokens, resulting in faster and more efficient pretraining with strong downstream performance.

- [**ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**](https://arxiv.org/abs/2003.10555)<br />
  _Authors_: Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning<br />
  _Description_: Introduces ELECTRA, a model that replaces the generator-discriminator setup in pretraining. It achieves higher efficiency compared to BERT while maintaining strong performance on NLP tasks.

- [**An Analysis of ELECTRA for Sentiment Classification**](https://www.tandfonline.com/doi/abs/10.1080/09540091.2021.1985968)<br />
  _Authors_: S. Zhang, H. Yu, G. Zhu<br />
  _Description_: Explores ELECTRA’s application in sentiment classification of Chinese text, emphasizing its efficiency in handling short comments.

- [**ELECTRA-Based Neural Coreference Resolution**](https://ieeexplore.ieee.org/abstract/document/9826714/)<br />

  _Authors_: F. Gargiulo, A. Minutolo, R. Guarasci, E. Damiano<br />
  _Description_: Leverages ELECTRA for coreference resolution tasks, demonstrating its potential in improving co-reference accuracy in text.

- [**ELECTRA for Biomedical Named Entity Recognition**](https://arxiv.org/abs/2011.04659)<br />
  _Authors_: S. Wang, T. Zhang<br />
  _Description_: Adapts ELECTRA for biomedical text processing, focusing on named entity recognition in domain-specific corpora.

- [**Fine-Tuning ELECTRA for Efficient Text Summarization**](https://aclanthology.org/2022.naacl-main.230/)<br />
  _Authors_: A. Banerjee, L. White<br />
  _Description_: Presents fine-tuning methods for ELECTRA to improve its performance on text summarization tasks efficiently.

### **9.10 XLNet**

**XLNet** is a transformer-based model that addresses the limitations of BERT by leveraging a permutation-based training objective. This allows XLNet to capture bidirectional context while avoiding the masking limitations of BERT, resulting in improved performance on various NLP tasks.

- [**XLNet: Generalized Autoregressive Pretraining for Language Understanding**](https://arxiv.org/abs/1906.08237)<br />
  _Authors_: Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le<br />
  _Description_: Introduces XLNet, which integrates autoregressive and autoencoding objectives to overcome limitations in BERT. It uses permutation-based training to improve context understanding."

- [**XLNet for Text Classification**](https://www.mdpi.com/2076-3417/12/18/8983)<br />
  _Authors_: F. Shi, S. Kai, J. Zheng, Y. Zhong<br />
  _Description_: Explores fine-tuning XLNet for text classification tasks, demonstrating significant improvements over baseline models."

- [**Comparing XLNet and BERT for Computational Characteristics**](https://scale.snu.ac.kr/papers/2020-01-Conference-ICEIC-BERTvs.XLNet.pdf)<br />
  _Authors_: H. Li, J. Choi, S. Lee, J.H. Ahn<br />
  _Description_: Compares XLNet and BERT from the perspective of computational efficiency, emphasizing training speed and resource utilization."

- [**XLNet-CNN: Combining Global Context with Local Context for Text Classification**](https://dl.acm.org/doi/pdf/10.1145/3704522.3704540)<br />
  _Authors_: A. Shahriar, D. Pandit, M.S. Rahman<br />
  _Description_: Combines XLNet with convolutional neural networks to capture both global and local contexts, enhancing text classification accuracy."

- [**DialogXL: Emotion Recognition in Conversations**](https://ojs.aaai.org/index.php/AAAI/article/view/17625)<br />
  _Authors_: W. Shen, J. Chen, X. Quan, Z. Xie<br />
  _Description_: Proposes DialogXL, an extended XLNet framework tailored for emotion recognition in multi-party conversations."

### **9.11 BERTweet**

**BERTweet** is a transformer model specifically pre-trained on a large corpus of English tweets. It is optimized for tasks in the social media domain, such as sentiment analysis, hate speech detection, and user intent classification.

- [**BERTweet: A Pre-trained Language Model for English Tweets**](https://arxiv.org/abs/2005.10200)</br>
  _Authors_: DQ Nguyen, T Vu, AT Nguyen</br>
  _Description_: Introduces BERTweet, the first large-scale language model pre-trained on English tweets, showcasing its effectiveness in social media text analysis.

- [**Classifying Tweet Sentiment Using the Hidden State and Attention Matrix of a Fine-tuned BERTweet Model**](https://arxiv.org/abs/2109.14692)</br>
  _Authors_: T. Macrì, F. Murphy, Y. Zou, Y. Zumbach</br>
  _Description_: Explores BERTweet's ability to classify tweet sentiments, utilizing its hidden states and attention matrices for enhanced accuracy.

- [**BERTweet.BR: A Pre-trained Language Model for Tweets in Portuguese**](https://link.springer.com/article/10.1007/s00521-024-10711-3)</br>
  _Authors_: F. Carneiro, D. Vianna, J. Carvalho, A. Plastino</br>
  _Description_: Adapts BERTweet for Portuguese tweets, highlighting its multilingual capabilities in processing social media text.

- [**Enhancing Health Tweet Classification: An Evaluation of Transformer-Based Models for Comprehensive Analysis**](https://search.proquest.com/openview/46061ed8eb5a5d1c006942caa5376688/1?pq-origsite=gscholar&cbl=18750&diss=y)</br>
  _Authors_: F.P. Patel</br>
  _Description_: Evaluates the use of BERTweet for health-related tweet classification, achieving notable improvements through BiLSTM augmentation.

- [**A BERTweet-Based Design for Monitoring Behavior Change Based on Five Doors Theory on Coral Bleaching Campaign**](https://link.springer.com/article/10.1186/s40537-022-00615-1)</br>
  _Authors_: G.N. Harywanto, J.S. Veron, D. Suhartono</br>
  _Description_: Leverages BERTweet to monitor behavioral changes in social media campaigns, utilizing the Five Doors Theory framework.

### **9.12 BlenderBot**

**BlenderBot**, developed by Facebook AI, is an open-domain chatbot capable of engaging in human-like conversations. It combines the conversational abilities of retrieval-based models with generative approaches, enabling it to generate more contextually appropriate and engaging responses.

- [**BlenderBot: Towards a More Open-Domain, Conversational AI Model**](https://arxiv.org/abs/2004.13637)</br>
  _Authors_: Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, Jason Weston</br>
  _Description_: Introduces BlenderBot, an open-domain chatbot designed to deliver engaging and knowledgeable conversations by fine-tuning conversational datasets with enhanced generative capabilities.

- [**BlenderBot 3: A Conversational Agent for Responsible Engagement**](https://arxiv.org/abs/2208.03188)</br>
  _Authors_: Kurt Shuster, Jing Xu, Morteza Komeili, Emily Smith, Jason Weston</br>
  _Description_: Details the advancements in BlenderBot 3, focusing on continual learning, safety mechanisms, and the model’s ability to adapt to user feedback in real-time.

- [**Empirical Analysis of BlenderBot 2.0 for Open-Domain Conversations**](https://arxiv.org/abs/2201.03239)</br>
  _Authors_: J Lee, M Shim, S Son, Y Kim, H Lim</br>
  _Description_: Examines the shortcomings of BlenderBot 2.0 across model, data, and user-centric approaches, offering insights for improvements in future iterations.

- [**GE-Blender: Graph-Based Knowledge Enhancement for Blender**](https://arxiv.org/abs/2301.12850)</br>
  _Authors_: X Lian, X Tang, Y Wang</br>
  _Description_: Proposes a graph-based knowledge-enhancement framework to improve BlenderBot’s ability to provide more accurate and contextually enriched responses.

- [**Enhancing Commonsense Knowledge in BlenderBot**](https://www.mdpi.com/1999-5903/15/12/384)</br>
  _Authors_: O Kobza, D Herel, J Cuhel, T Gargiani, J Pichl, P Marek</br>
  _Description_: Explores methods to augment commonsense knowledge in BlenderBot, improving conversational consistency and user engagement.

### **9.13 DeBERTa**

**DeBERTa (Decoding-enhanced BERT with Disentangled Attention)** improves upon BERT and RoBERTa by introducing disentangled attention mechanisms and an enhanced mask decoder. These innovations allow DeBERTa to achieve state-of-the-art results on a variety of NLP benchmarks.

- [**DeBERTa: Decoding-Enhanced BERT with Disentangled Attention**](https://arxiv.org/abs/2006.03654)</br>
  _Authors_: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen</br>
  _Description_: Introduces DeBERTa, which improves upon BERT by using disentangled attention and a novel position encoding mechanism, achieving state-of-the-art results across multiple NLP benchmarks.

- [**DeBERTa-v3: Improving DeBERTa Using ELECTRA-Style Pre-Training**](https://arxiv.org/abs/2111.09543)</br>
  _Authors_: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen</br>
  _Description_: Builds on DeBERTa with ELECTRA-style pretraining and gradient-disentangled embedding sharing, enhancing performance and training efficiency.

- [**Therapeutic Prediction Task on Electronic Health Record Using DeBERTa**](
  "link": "https://www.researchgate.net/profile/Arti-Gupta-6/publication/366664269_Therapeutic_Prediction_task_on_Electronic_Health_Record_using_DeBERTa/links/63e88c56dea6121757a25ef7ic-Prediction-task-on-Electronic-Health-Record-using-DeBERTa.pdf)</br>
  _Authors_: A. Gupta, V.K. Chaurasiya</br>
  _Description_: Applies DeBERTa to predict therapeutic outcomes in electronic health records, demonstrating its utility in domain-specific NLP tasks.

- [**Aspect Sentiment Classification via Local Context-Focused Syntax Based on DeBERTa**](https://ieeexplore.ieee.org/abstract/document/9986398/)</br>
  _Authors_: J. Liu, Z. Zhang, X. Lu</br>
  _Description_: Proposes a local context-focused syntax method using DeBERTa for aspect-based sentiment classification, achieving notable improvements.

- [**A Novel DeBERTa-Based Model for Financial Question Answering**](https://arxiv.org/abs/2207.05875)</br>
  _Authors_: Y.J. Wang, Y. Li, H. Qin, Y. Guan, S. Chen</br>
  _Description_: Develops a DeBERTa-based approach for answering financial questions, incorporating optimization techniques for improved accuracy.

### **9.14 BigBird**

**BigBird** is a sparse attention transformer designed to handle long sequences efficiently. It is particularly useful for tasks involving long documents, such as summarization and question answering, where standard transformers struggle due to memory constraints.

- [**Big Bird: Transformers for Longer Sequences**](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)</br>
  _Authors_: Manzil Zaheer, Guru Guruganesh, Kaushik Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed</br>
  _Description_: This paper introduces BigBird, a transformer model designed for efficient handling of longer sequences using a sparse attention mechanism, reducing computational complexity from quadratic to linear.

- [**ICDBigBird: A Contextual Embedding Model for ICD Code Classification**](https://arxiv.org/abs/2204.10408)</br>
  _Authors_: G. Michalopoulos, M. Malyska, N. Sahar, A. Wong</br>
  _Description_: Proposes a BigBird-based contextual embedding model tailored for ICD code classification in medical records, showcasing the model's capacity for domain-specific applications.

- [**Clinical-longformer and Clinical-BigBird: Transformers for Long Clinical Sequences**](https://arxiv.org/abs/2201.11838)</br>
  _Authors_: Y. Li, R. Wehbe, F. Ahmad, H. Wang, Y. Luo</br>
  _Description_: Develops Clinical-BigBird for processing long clinical text sequences, highlighting its performance improvements compared to other transformer models.

- [**Attention-Free BigBird Transformer for Long Document Text Summarization**](https://cspub-ijcisim.org/index.php/ijcisim/article/download/633/589)</br>
  _Authors_: G. Mishra, N. Sethi, A. Loganathan</br>
  _Description_: Introduces a modified BigBird transformer for document summarization, removing attention-based mechanisms for better efficiency.

- [**Vision BigBird: Random Sparsification for Full Attention**](https://arxiv.org/abs/2311.05988)</br>
  _Authors_: Z. Zhang, X. Gong</br>
  _Description_: Applies BigBird concepts to vision transformers, proposing a random sparsification mechanism to optimize full attention for vision tasks.

### **9.15 PEGASUS**

**PEGASUS** is a transformer model developed for abstractive summarization tasks. It uses a novel pretraining objective called "Gap Sentences Generation" to better understand document structure and generate high-quality summaries.

- [**PEGASUS: Pre-training with Extracted Gap-Sentences for Abstractive Summarization**](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf)</br>
  _Authors_: Jingqing Zhang, Yao Zhao, Mohammad Saleh, Peter J. Liu</br>
  _Description_: This paper introduces PEGASUS, a model designed for abstractive summarization. It uses a novel pretraining objective, Gap Sentence Generation, to achieve state-of-the-art performance on multiple summarization tasks.

- [**Improving News Summarization with PEGASUS**](https://arxiv.org/pdf/2109.09272)</br>
  _Authors_: T. Yang, Z. Li, W. Zhang</br>
  _Description_: Explores the use of PEGASUS for news summarization, showcasing improvements in coherence and informativeness.

- [**Domain Adaptation of PEGASUS for Scientific Document Summarization**](https://dl.acm.org/doi/abs/10.1145/3477482)</br>
  _Authors_: R. Khan, S. Basu, J. Dutta</br>
  _Description_: Adapts PEGASUS for summarizing scientific documents, focusing on domain-specific challenges and evaluation metrics.

- [**Extractive and Abstractive Summarization with PEGASUS on Low-Resource Languages**](https://ieeexplore.ieee.org/abstract/document/9501123/)</br>
  _Authors_: A. Sharma, L. Wu, Y. Wang</br>
  _Description_: Applies PEGASUS for summarization tasks in low-resource languages, demonstrating its adaptability and potential in multilingual NLP.

- [**Analysis of Pretraining Objectives in PEGASUS**](https://arxiv.org/pdf/2106.05802)</br>
  _Authors_: M. Singh, J. Luo, X. Hu</br>
  _Description_: Investigates the impact of various pretraining objectives on the performance of PEGASUS, offering insights into optimization strategies.

### **9.16 FLAN-T5**

**FLAN-T5** is a fine-tuned version of T5 that incorporates instruction tuning across multiple NLP tasks. This makes it more versatile and capable of zero-shot or few-shot learning for new tasks, improving its generalization capabilities.

- [**The Flan Collection: Designing Data and Methods for Effective Instruction Tuning**](https://proceedings.mlr.press/v202/longpre23a.html)</br>
  _Authors_: S. Longpre, L. Hou, T. Vu, A. Webson</br>
  _Description_: Explores the design decisions enabling FLAN-T5 to outperform prior instruction-tuned models by significant margins, while requiring less fine-tuning to achieve optimal performance.

- [**A Zero-Shot and Few-Shot Study of Instruction-Finetuned Large Language Models Applied to Clinical and Biomedical Tasks**](https://arxiv.org/abs/2307.12114)</br>
  _Authors_: Y. Labrak, M. Rouvier, R. Dufour</br>
  _Description_: Examines FLAN-T5's performance in zero-shot and few-shot scenarios on biomedical tasks, highlighting its adaptability and robustness in domain-specific applications.

- [**Enhancing Amblyopia Identification Using NLP: A Study of BioClinical BERT and FLAN-T5 Models**](https://iovs.arvojournals.org/article.aspx?articleid=2794655)</br>
  _Authors_: W.C. Lin, C. Reznick, L. Reznick, A. Lucero</br>
  _Description_: Investigates the use of FLAN-T5 in identifying amblyopia-related conditions, emphasizing its application in clinical text processing.

- [**Semantic Feature Verification in FLAN-T5**](https://arxiv.org/abs/2304.05591)</br>
  _Authors_: S. Suresh, K. Mukherjee, T.T. Rogers</br>
  _Description_: Explores FLAN-T5's effectiveness in semantic feature verification tasks, comparing it with other models optimized for question-answering.

- [**Leveraging Distillation Techniques for Document Understanding: A Case Study with FLAN-T5**](https://arxiv.org/abs/2409.11282)</br>
  _Authors_: M. Lamott, M.A. Shakir</br>
  _Description_: Highlights the integration of distillation techniques with FLAN-T5 to improve document understanding in various NLP tasks.

### **9.17 MobileBERT**

**MobileBERT** is a compact version of BERT optimized for mobile and edge devices. It maintains strong performance on NLP tasks while being significantly smaller and faster, making it ideal for resource-constrained environments.

- [**MobileBERT: A Compact Task-Agnostic BERT for Resource-Limited Devices**](https://arxiv.org/abs/2004.02984)</br>
  _Authors_: Zhenzhong Sun, Hongyu Yu, Xiaodan Song, Renjie Liu, Yang Yang, Denny Zhou</br>
  _Description_: Introduces MobileBERT, a compact version of BERT designed for resource-limited devices. It uses knowledge distillation and carefully designed transformer blocks to achieve performance comparable to BERT while being computationally efficient.

- [**ICDBigBird and MobileBERT for Efficient Clinical Text Classification**](https://arxiv.org/abs/2204.10408)</br>
  _Authors_: G. Michalopoulos, M. Malyska, N. Sahar, A. Wong</br>
  _Description_: Applies MobileBERT in conjunction with other models to classify clinical text, highlighting its utility in low-resource and domain-specific environments.

- [**Quantized MobileBERT for Real-Time NLP Applications**](https://arxiv.org/abs/2310.03971)</br>
  _Authors_: S.S. Roy, S. Nilizadeh</br>
  _Description_: Explores quantization techniques to further enhance the deployment of MobileBERT in real-time edge devices.

- [**MobileBERT in Toxic Comment Classification Using Knowledge Distillation**](https://norma.ncirl.ie/id/eprint/6132)</br>
  _Authors_: Bijender Gupta</br>
  _Description_: Utilizes MobileBERT with knowledge distillation to classify toxic comments effectively, demonstrating its flexibility in social media text analysis.

- [**Real-Time Execution of MobileBERT on Mobile Devices**](https://arxiv.org/abs/2009.06823)</br>
  _Authors_: W. Niu, Z. Kong, G. Yuan, W. Jiang, J. Guan</br>
  _Description_: Examines MobileBERT's performance on mobile devices, focusing on optimizing real-time execution and deployment.

### **9.18 GPT-Neo**

**GPT-Neo** is an open-source alternative to GPT-3, developed by EleutherAI. It offers a similar architecture and is pre-trained on large datasets, enabling it to perform generative NLP tasks like text completion and summarization.

- [**GPT-Neo: An Open-Source Autoregressive Language Model**](https://arxiv.org/abs/2204.06745)</br>
  _Authors_: S Black, S Biderman, E Hallahan, Q Anthony, S Foster</br>
  _Description_: Presents GPT-Neo, an open-source alternative to proprietary autoregressive language models. It emphasizes community-driven development and large-scale model training.

- [**GPT-Neo for Commonsense Reasoning--A Theoretical and Practical Lens**](https://arxiv.org/pdf/2211.15593)</br>
  _Authors_: R Kashyap, V Kashyap</br>
  _Description_: Examines the performance of GPT-Neo in commonsense reasoning tasks, comparing it with other large language models and discussing theoretical implications.

- [**Enhancing Contextual Understanding in Large Language Models with GPT-Neo**](https://www.researchsquare.com/article/rs-4814991/latest)</br>
  _Authors_: M Ito, H Nishikawa, Y Sakamoto</br>
  _Description_: Explores improvements in GPT-Neo's contextual understanding using dynamic dependency structures in large-scale language models.

- [**Generating Fake Cyber Threat Intelligence Using GPT-Neo**](https://ieeexplore.ieee.org/abstract/document/10248596/)</br>
  _Authors_: Z Song, Y Tian, J Zhang, Y Hao</br>
  _Description_: Investigates the use of GPT-Neo for generating fake cyber threat intelligence, showcasing its capabilities and potential risks.

- [**Evaluating the Carbon Impact of Large Language Models: GPT-Neo**](https://ieeexplore.ieee.org/abstract/document/10253886/)</br>
  _Authors_: B Everman, T Villwock, D Chen, N Soto</br>
  _Description_: Analyzes the carbon footprint of GPT-Neo during inference, highlighting the environmental implications of deploying large-scale language models.

### **9.19 Longformer**

**Longformer** addresses the limitations of standard transformers with sparse attention, enabling it to process long sequences efficiently. It is suitable for tasks like document classification, summarization, and long-context question answering.

- [**Longformer: The Long-Document Transformer**](https://arxiv.org/abs/2004.05150)</br>
  _Authors_: Iz Beltagy, Matthew E. Peters, Arman Cohan</br>
  _Description_: This paper introduces Longformer, a transformer model optimized for long documents. It uses a sparse attention mechanism that scales linearly with sequence length, making it suitable for processing thousands of tokens efficiently.

- [**Long Range Arena: A Benchmark for Efficient Transformers**](https://arxiv.org/abs/2011.04006)</br>
  _Authors_: Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler</br>
  _Description_: Provides a systematic benchmark to evaluate transformer models, including Longformer, for long-range attention tasks, emphasizing efficiency and performance.

- [**Longformer for Multi-Document Summarization**](https://www.mdpi.com/2079-9292/11/11/1706)</br>
  _Authors_: F. Yang, S. Liu</br>
  _Description_: Applies Longformer to extractive summarization of multiple documents, showcasing its ability to handle large-scale text summarization tasks effectively.

- [**Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding**](
  "link": "https://openaccess.thecvf.com/content/ICCV2021/papersti-Scale_Vision_Longformer_A_New_Vision_Transformer_for_High-Resolution_Image_ICCV_2021_paper.pdf)</br>
  _Authors_: P. Zhang, X. Dai, J. Yang</br>
  _Description_: Adapts Longformer concepts for vision tasks, focusing on encoding high-resolution images with sparse attention for computational efficiency.

- [**Longformer for Dense Document Retrieval**](https://aclanthology.org/2023.emnlp-main.223.pdf)</br>
  _Authors_: J. Yang, Z. Liu, G. Sun</br>
  _Description_: Explores Longformer as a dense document retrieval model, demonstrating its ability to process and retrieve information from long-form text effectively.

### **9.20 XLM-RoBERTa**

**XLM-RoBERTa** is a multilingual variant of RoBERTa designed to handle over 100 languages. It is highly effective for cross-lingual understanding tasks, such as translation and multilingual question answering.

- [**Unsupervised Cross-lingual Representation Learning at Scale**](https://arxiv.org/abs/1911.02116)</br>
  _Authors_: Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov</br>
  _Description_: Introduces XLM-RoBERTa, a multilingual model pre-trained on 100 languages. It achieves state-of-the-art results in cross-lingual understanding tasks and is fine-tuned for various multilingual NLP applications.

- [**A Conspiracy Theory Text Detection Method Based on RoBERTa and XLM-RoBERTa Models**](https://downloads.webis.de/pan/publications/papers/zeng_2024.pdf)</br>
  _Authors_: Z. Zeng, Z. Han, J. Ye, Y. Tan, H. Cao, Z. Li</br>
  _Description_: Combines XLM-RoBERTa and RoBERTa models for detecting conspiracy theories, with emphasis on multilingual applications.

- [**Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT**](https://ceur-ws.org/Vol-3497/paper-085.pdf)</br>
  _Authors_: H. Mohammadi, A. Giachanou, A. Bagheri</br>
  _Description_: Leverages XLM-RoBERTa for online sexism detection, demonstrating its effectiveness in multilingual contexts.

- [**Fine-tuning BERT, DistilBERT, XLM-RoBERTa, and Ukr-RoBERTa for Sentiment Analysis of Ukrainian Language Reviews**](https://jai.in.ua/archive/2024/2024-2-7.pdf)</br>
  _Authors_: M. Prytula</br>
  _Description_: Adapts XLM-RoBERTa for sentiment analysis of Ukrainian text, highlighting its cross-lingual capabilities.

- [**NER in Hindi Language Using Transformer Model: XLM-RoBERTa**](https://ieeexplore.ieee.org/abstract/document/9935841/)</br>
  _Authors_: A. Choure, R.B. Adhao</br>
  _Description_: Utilizes XLM-RoBERTa for named entity recognition in Hindi, showcasing its performance in low-resource languages.

### **9.21 DialoGPT**

**DialoGPT**, developed by Microsoft, is a conversational version of GPT-2 fine-tuned on dialogue datasets. It is designed to generate engaging, context-aware conversational responses for chatbots and other interactive applications.

- [**DialoGPT: Large-Scale Generative Pre-training for Dialogue**](https://arxiv.org/abs/1911.00536)</br>
  _Authors_: Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan</br>
  _Description_: DialoGPT extends GPT-2 for conversational AI by fine-tuning on large-scale dialogue datasets. It achieves state-of-the-art results in open-domain dialogue generation with engaging and coherent outputs.

- [**Småprat: DialogGPT for Natural Language Generation of Swedish Dialogue by Transfer Learning**](https://arxiv.org/abs/2110.06273)</br>
  _Authors_: T. Adewumi, R. Brännvall, N. Abid, M. Pahlavan</br>
  _Description_: Applies DialoGPT for Swedish dialogue generation, showcasing the model's adaptability to new languages through transfer learning.

- [**Augpt: Dialogue with Pre-Trained Language Models and Data Augmentation**](https://www.researchgate.net/publication/349195551)</br>
  _Authors_: J. Kulhánek, V. Hudecek, T. Nekvinda</br>
  _Description_: Enhances DialoGPT’s conversational capabilities with data augmentation techniques for multi-domain task-oriented dialogue systems.

- [**Generating Emotional Responses with DialoGPT-Based Multi-Task Learning**](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_38)</br>
  _Authors_: S. Cao, Y. Jia, C. Niu, H. Zan, Y. Ma</br>
  _Description_: Introduces a multi-task learning architecture for DialoGPT to generate emotionally grounded responses in conversations.

- [**On the Generation of Medical Dialogues for COVID-19 Using DialoGPT**](https://arxiv.org/abs/2005.05442)</br>
  _Authors_: W. Yang, G. Zeng, B. Tan, Z. Ju</br>
  _Description_: Explores DialoGPT for generating medical dialogues related to COVID-19, demonstrating its effectiveness in healthcare applications.

### **9.22 MarianMT**

**MarianMT** is a neural machine translation model developed by Facebook. It supports many language pairs and is optimized for low-resource languages, making it an excellent tool for translation tasks.

- [**Marian: Fast Neural Machine Translation in C++**](https://arxiv.org/abs/1804.00344)</br>
  _Authors_: J. Hieber, T. Domhan, M. Denkowski, D. Vilar, X. Wang, S. Fikri Aji, A. Clifton, M. Post</br>
  _Description_: Introduces MarianMT, a fast and efficient neural machine translation framework implemented in C++, optimized for production-scale translation tasks with high speed and accuracy.

- [**University of Amsterdam at the CLEF 2024 Joker Track**](https://ceur-ws.org/Vol-3740/paper-181.pdf)</br>
  _Authors_: E. Schuurman, M. Cazemier, L. Buijs</br>
  _Description_: Presents an application of MarianMT for multilingual machine translation tasks, highlighting its performance in competitive evaluation tracks.

- [**Controllability for English-Ukrainian Machine Translation Based on Specialized Corpora**](https://aclanthology.org/2023.multi3generation-1.1.pdf)</br>
  _Authors_: D. Maksymenko, O. Turuta, N. Saichyshyna</br>
  _Description_: Explores methods to enhance controllability in machine translation using MarianMT, focusing on adapting translation outputs to specific requirements.

- [**MarianCG: A Code Generation Transformer Model Inspired by Machine Translation**](https://link.springer.com/article/10.1186/s44147-022-00159-4)</br>
  _Authors_: A. Soliman, M. Hadhoud, S. Shaheen</br>
  _Description_: Demonstrates the versatility of MarianMT for tasks beyond language translation, including code generation.

- [**A Novel Effective Combinatorial Framework for Sign Language Translation**](https://ieeexplore.ieee.org/abstract/document/10109931/)</br>
  _Authors_: S. Lin, J. You, Z. He, H. Jia, L. Chen</br>
  _Description_: Uses MarianMT in a hybrid framework for translating sign language into text, emphasizing its adaptability to multimodal input.

### **9.23 Falcon**

**Falcon** is an open-source generative language model known for its lightweight architecture and efficient training. It is particularly useful for generating text with constrained computational resources.

- [**The Falcon Series of Open Language Models**](https://arxiv.org/abs/2311.16867)</br>
  _Authors_: E. Almazrouei, H. Alobeidli, A. Alshamsi</br>
  _Description_: This paper introduces the Falcon language models, emphasizing pretraining on large-scale datasets to deliver superior performance in generative and comprehension tasks.

- [**Falcon: Faster and Parallel Inference of Large Language Models**](https://arxiv.org/abs/2412.12639)</br>
  _Authors_: X. Gao, W. Xie, Y. Xiang, F. Ji</br>
  _Description_: Proposes a speculative decoding framework for Falcon models, designed to enhance inference speed and output quality through semi-autoregressive drafting.

- [**Falcon 2.0: An Entity and Relation Linking Tool over Wikidata**](https://dl.acm.org/doi/10.1145/3340531.3412777)</br>
  _Authors_: A. Sakor, K. Singh, A. Patel, M.E. Vidal</br>
  _Description_: Presents Falcon 2.0, a resource for linking entities and relations to Wikidata, optimized for applications requiring structured data linking.

- [**FALCON: A New Approach for the Evaluation of Opportunistic Networks**](https://www.sciencedirect.com/science/article/pii/S1570870518304530)</br>
  _Authors_: E. Hernández-Orallo, J.C. Cano, C.T. Calafate, P. Manzoni</br>
  _Description_: Develops FALCON as a model for evaluating the performance and scalability of opportunistic networks using advanced simulation techniques.

- [**Falcon: Rapid Statistical Fault Coverage Estimation for Complex Designs**](https://ieeexplore.ieee.org/document/6401584)</br>
  _Authors_: S. Mirkhani, J.A. Abraham</br>
  _Description_: Introduces a statistical model to estimate fault coverage in complex design architectures using the Falcon framework.

### **9.24 CodeGen**

**CodeGen** is a transformer model optimized for code generation tasks. It has been fine-tuned on programming-related datasets, enabling it to write code snippets in languages like Python, JavaScript, and more.

- [**Codereval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models**](https://arxiv.org/pdf/2302.00288)</br>
  _Authors_: H. Yu, B. Shen, D. Ran, J. Zhang, Q. Zhang, Y. Ma</br>
  _Description_: Presents a comprehensive benchmark evaluating CodeGen and similar models for practical code generation tasks, emphasizing pretraining on domain-specific data.

- [**Deep Learning for Source Code Modeling and Generation**](https://dl.acm.org/doi/abs/10.1145/3383458)</br>
  _Authors_: T.H.M. Le, H. Chen, M.A. Babar</br>
  _Description_: Analyzes deep learning techniques, including CodeGen, for source code generation and modeling, addressing applications and challenges in the field.

- [**Teaching Code LLMs to Use Autocompletion Tools in Repository-Level Code Generation**](https://arxiv.org/pdf/2401.06391)</br>
  _Authors_: C. Wang, J. Zhang, Y. Feng, T. Li, W. Sun, Y. Liu</br>
  _Description_: Introduces techniques for enhancing CodeGen’s performance using repository-level data and autocompletion tools.

- [**CodeGen-Search: A Code Generation Model Incorporating Similar Sample Information**](https://www.worldscientific.com/doi/abs/10.1142/S0218194023500584)</br>
  _Authors_: H.W. Li, J.L. Kuang, M.S. Zhong, Z.X. Wang</br>
  _Description_: Proposes a variant of CodeGen integrating similar sample information to improve accuracy in code generation.

- [**CodeP: Grammatical Seq2Seq Model for General-Purpose Code Generation**](https://dl.acm.org/doi/abs/10.1145/3597926.3598048)</br>
  _Authors_: Y. Dong, G. Li, Z. Jin</br>
  _Description_: Explores grammar-based improvements to CodeGen for enhancing its general-purpose code generation capabilities.

### **9.25 ByT5**

**ByT5** is a byte-level version of the T5 model. It eliminates the need for tokenization by processing raw byte inputs, making it especially effective for multilingual tasks and handling unseen text encodings.

- [**ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models**](https://direct.mit.edu/tacl/article-abstract/doi/10.1162/tacl_a_00461/110049)</br>
  _Authors_: L Xue, A Barua, N Constant, R Al-Rfou</br>
  _Description_: Introduces ByT5, a token-free pre-trained model that processes text directly as raw bytes. This novel approach eliminates tokenization, enabling better handling of rare and unseen text.

- [**Post-OCR Correction of Digitized Swedish Newspapers with ByT5**](https://aclanthology.org/2024.latechclfl-1.23/)</br>
  _Authors_: V Löfgren, D Dannélls</br>
  _Description_: Explores the use of ByT5 for correcting OCR errors in digitized historical Swedish newspapers, highlighting its ability to generalize across noisy text.

- [**One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit NLP Tasks**](https://arxiv.org/abs/2409.13920)</br>
  _Authors_: S Nehrdich, O Hellwig, K Keutzer</br>
  _Description_: Adapts ByT5 for Sanskrit NLP tasks, showcasing its flexibility in handling morphologically rich languages with byte-level encoding.

- [**Fine-Tashkeel: Fine-Tuning Byte-Level Models for Accurate Arabic Text Diacritization**](https://ieeexplore.ieee.org/abstract/document/10185725/)</br>
  _Authors_: B Al-Rfooh, G Abandah</br>
  _Description_: Applies ByT5 to Arabic text diacritization, demonstrating its effectiveness in handling the intricacies of script-based languages.

- [**Tokenization and Morphology in Multilingual Language Models: A Comparative Analysis of mT5 and ByT5**](https://arxiv.org/abs/2410.11627)</br>
  _Authors_: TA Dang, L Raviv, L Galke</br>
  _Description_: Compares ByT5 and mT5 in multilingual tasks, emphasizing the advantages of byte-level processing for languages with complex morphology.

### **9.26 PhoBERT**

**PhoBERT** is a pre-trained language model tailored for Vietnamese. It is optimized for NLP tasks in Vietnamese, such as sentiment analysis, text classification, and named entity recognition.

- [**PhoBERT: Pre-trained language models for Vietnamese**](https://arxiv.org/abs/2003.00744)</br>
  _Authors_: Dung Quoc Nguyen, Anh Tuan Nguyen</br>
  _Description_: Introduces PhoBERT, the first large-scale monolingual BERT-based language model pre-trained for Vietnamese. It outperforms multilingual models on various Vietnamese NLP tasks, highlighting the importance of monolingual pretraining.

- [**Stock Article Title Sentiment-Based Classification Using PhoBERT**](https://ceur-ws.org/Vol-3026/paper25.pdf)</br>
  _Authors_: NS Tun, NN Long, T Tran, NT Thao</br>
  _Description_: Utilizes PhoBERT for sentiment classification of stock-related article titles, demonstrating its effectiveness in financial text analysis.

- [**PhoBERT: Application in Disease Classification Based on Vietnamese Symptom Analysis**](https://sciendo.com/pdf/10.2478/acss-2023-0004)</br>
  _Authors_: HT Nguyen, TN Huynh, NTN Mai, KDD Le</br>
  _Description_: Applies PhoBERT to classify diseases from Vietnamese symptom descriptions, showcasing its adaptability for medical NLP tasks.

- [**A Text Classification for Vietnamese Feedback via PhoBERT-Based Deep Learning**](https://link.springer.com/chapter/10.1007/978-981-19-2394-4_24)</br>
  _Authors_: CV Loc, TX Viet, TH Viet, LH Thao, NH Viet</br>
  _Description_: Proposes a PhoBERT-based deep learning framework for Vietnamese text classification tasks, improving performance on customer feedback analysis.

- [**Fine-Tuned PhoBERT for Sentiment Analysis of Vietnamese Phone Reviews**](https://ctujs.ctu.edu.vn/index.php/ctujs/article/view/1146)</br>
  _Authors_: TM Ngo, BH Ngo, SV Valerievich</br>
  _Description_: Examines the application of PhoBERT for sentiment analysis on Vietnamese phone reviews, focusing on fine-tuning techniques.

### **9.27 Funnel Transformer**

**Funnel Transformer** introduces a pooling mechanism to reduce the computational complexity of transformers. This hierarchical approach improves scalability while maintaining performance for long-sequence tasks.

- [**Funnel-Transformer: Filtering Out Sequential Redundancy for Efficient Language Processing**](https://proceedings.neurips.cc/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf)</br>
  _Authors_: Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le</br>
  _Description_: This paper introduces Funnel Transformer, which reduces computational redundancy in sequence processing through a funnel-shaped architecture. It balances efficiency and performance in language understanding tasks.

- [**Do Transformer Modifications Transfer Across Implementations and Applications?**](https://arxiv.org/abs/2102.11972)</br>
  _Authors_: Srinivasan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry</br>
  _Description_: Analyzes various transformer modifications, including Funnel Transformer, to evaluate their adaptability and performance across applications.

- [**Condenser: A Pre-training Architecture for Dense Retrieval**](https://arxiv.org/abs/2104.08253)</br>
  _Authors_: Linfeng Gao, Jianfeng Callan</br>
  _Description_: Explores Condenser, a variant inspired by Funnel Transformer, optimized for dense text retrieval tasks with enhanced efficiency.

- [**ArabicTransformer: Efficient Large Arabic Language Model with Funnel Transformer**](https://aclanthology.org/2021.findings-emnlp.108)</br>
  _Authors_: Saad Alrowili, K. Vijay-Shanker</br>
  _Description_: Adapts Funnel Transformer for Arabic NLP tasks, focusing on improving efficiency while maintaining accuracy for resource-intensive language models.

- [**Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator**](https://arxiv.org/abs/2305.15099)</br>
  _Authors_: Ziyang He, Ming Feng, Jun Leng</br>
  _Description_: Proposes Fourier Transformer, inspired by Funnel Transformer, for efficient modeling of long-range dependencies using Fourier transforms.

### **9.28 T5v1.1**

**T5v1.1** is an improved version of the original T5 model. It features architectural changes and optimizations, resulting in enhanced performance and better efficiency for a wide range of NLP tasks.

- [**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](https://arxiv.org/abs/1910.10683)</br>
  _Authors_: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu</br>
  _Description_: This foundational paper introduces the T5 framework, which forms the basis for T5v1.1. It treats all NLP tasks as a text-to-text problem, enabling seamless multitask learning and fine-tuning.

- [**Improved Fine-Tuning and Parameter Sharing in T5 Models**](https://arxiv.org/abs/2205.10696)</br>
  _Authors_: V. Lialin, K. Zhao, N. Shivagunde</br>
  _Description_: Proposes refinements for the T5 architecture, including T5v1.1, focusing on enhanced parameter sharing and optimized fine-tuning strategies.

- [**T5v1.1 for Low-Resource Language Understanding**](https://aclanthology.org/2023.emnlp-main.337/)</br>
  _Authors_: D. Mehra, L. Xie, E. Hofmann-Coyle</br>
  _Description_: Explores the use of T5v1.1 in low-resource language tasks, demonstrating its ability to adapt and perform well on limited data.

- [**Enhanced Dialogue State Tracking Using T5v1.1**](https://arxiv.org/abs/2305.17020)</br>
  _Authors_: P. Lesci, Y. Fujinuma, M. Hardalov, C. Shang</br>
  _Description_: Demonstrates the efficiency of T5v1.1 for dialogue state tracking tasks, leveraging its text-to-text capabilities for complex conversational scenarios.

- [**T5v1.1 in Scientific Document Summarization**](https://aclanthology.org/2024.repl4nlp-1.9/)</br>
  _Authors_: R. Uppaal, Y. Li, J. Hu</br>
  _Description_: Applies T5v1.1 for summarizing scientific documents, emphasizing its superior abstractive summarization performance compared to baseline models.

### **9.29 RoFormer**

**RoFormer (Rotary Position Embeddings Transformer)** incorporates rotary position embeddings to improve positional encoding in transformers. This innovation enhances its capability to handle longer sequences and tasks like language modeling and translation.

- [**RoFormer: Enhanced Transformer with Rotary Position Embedding**](https://arxiv.org/abs/2104.09864)</br>
  _Authors_: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, Y. Liu</br>
  _Description_: Introduces RoFormer, a transformer model with rotary position embeddings designed to efficiently handle positional information. It improves performance across tasks requiring long-range dependencies.

- [**RoFormer for Position-Aware Multiple Instance Learning in Whole Slide Image Classification**](https://link.springer.com/chapter/10.1007/978-3-031-45676-3_44)</br>
  _Authors_: E. Pochet, R. Maroun, R. Trullo</br>
  _Description_: Adapts RoFormer for position-aware multiple instance learning in medical image classification, emphasizing its flexibility for multimodal tasks.

- [**RoGraphER: Enhanced Extraction of Chinese Medical Entity Relationships Using RoFormer**](https://www.mdpi.com/2079-9292/13/15/2892)</br>
  _Authors_: Q. Zhang, Y. Sun, P. Lv, L. Lu, M. Zhang, J. Wang, C. Wan</br>
  _Description_: Leverages RoFormer for extracting medical entity relationships, showcasing its application in healthcare NLP tasks.

- [**Chinese Event Extraction Method Based on RoFormer**](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2023/8268651)</br>
  _Authors_: B. Qiang, X. Zhou, Y. Wang, X. Yang</br>
  _Description_: Presents a Chinese event extraction framework using RoFormer with FGM and CRF for enhanced performance.

- [**Entity Linking Based on RoFormer-Sim for Chinese Short Texts**](https://drpress.org/ojs/index.php/fcis/article/view/9422)</br>
  _Authors_: W. Xie</br>
  _Description_: Proposes an entity linking model based on RoFormer-Sim for improving accuracy in Chinese short-text processing.

### **9.30 MBart and MBart-50**

**MBart (Multilingual BART)** and its extension **MBart-50** are encoder-decoder models optimized for multilingual tasks, including translation across 50 languages. They are pre-trained on large-scale multilingual data and fine-tuned for tasks like summarization and language generation.

- [**mBART: Multilingual Denoising Pretraining for Neural Machine Translation**](https://arxiv.org/abs/2001.08210)</br>
  _Authors_: Tang, Yuqing, Angela Fan, Mikel Artetxe, Seta Celikyilmaz, Yulia Tsvetkov, Luke Zettlemoyer, Veselin Stoyanov</br>
  _Description_: This foundational paper introduces mBART, a multilingual sequence-to-sequence model pre-trained with denoising objectives. It demonstrates strong performance on machine translation and cross-lingual tasks.

- [**mBART-50: Multilingual Translation with a Fine-Tuned mBART Model**](https://arxiv.org/abs/2008.00401)</br>
  _Authors_: Tang, Yuqing, Chau Tran, Xian Li, Angela Fan, Dmytro Okhonko, Edouard Grave</br>
  _Description_: Presents mBART-50, an extension of mBART pre-trained on 50 languages. It achieves state-of-the-art performance in zero-shot translation tasks.

- [**Fine-Tuning mBART for Low-Resource Machine Translation**](https://aclanthology.org/2021.wat-1.23/)</br>
  _Authors_: R. Dabre, A. Chakrabarty</br>
  _Description_: Discusses fine-tuning techniques for mBART on Indic languages, showing significant improvements in low-resource translation scenarios.

- [**ZmBART: An Unsupervised Cross-Lingual Transfer Framework for Language Generation**](https://arxiv.org/abs/2106.01597)</br>
  _Authors_: K. K. Maurya, M. S. Desarkar, Y. Kano</br>
  _Description_: Proposes ZmBART, a variant of mBART adapted for unsupervised cross-lingual generation, highlighting its potential for broader NLP applications.

- [**Fine-Tuning mBART-50 for Domain-Specific Neural Machine Translation**](https://aclanthology.org/2023.mtsummit-users.14/)</br>
  _Authors_: B. Namdarzadeh, S. Mohseni, L. Zhu</br>
  _Description_: Explores the application of mBART-50 for domain-specific translations, such as legal and medical text, showcasing its adaptability.

- [**DMSeqNet-mBART: Enhancing mBART for Chinese Short News Text Summarization**](https://www.sciencedirect.com/science/article/pii/S0957417424019626)</br>
  _Authors_: K. Cao, Y. Hao, W. Cheng</br>
  _Description_: Presents DMSeqNet-mBART, a specialized adaptation of mBART for summarizing Chinese short news, enhancing performance on specific linguistic challenges.

- [**Cross-Lingual Reverse Dictionary Using Multilingual mBART**](https://link.springer.com/chapter/10.1007/978-981-99-1912-3_11)</br>
  _Authors_: A. Mangal, S. S. Rathore, K. V. Arya</br>
  _Description_: Demonstrates the use of mBART for cross-lingual reverse dictionary tasks, highlighting its effectiveness in multilingual semantic understanding.

## **10. Datasets**

Datasets play a crucial role in training and evaluating NLP models. The choice of dataset depends on the specific NLP task, as different datasets cater to different use cases, such as text generation, classification, named entity recognition, question answering, and more. Below, we provide a categorized list of commonly used datasets for various NLP tasks.

### **10.1 Text Generation Datasets**

These datasets are used to train models that generate coherent and contextually relevant text based on a given input. Common applications include dialogue systems, story generation, and code completion.

- [**Scigen: A Dataset for Reasoning-Aware Text Generation from Scientific Tables**](https://openreview.net/forum?id=Jul-uX7EV_I)</br>
  _Authors_: N.S. Moosavi, A. Rücklé, D. Roth</br>
  _Description_: Introduces SciGen, a dataset designed for text generation tasks requiring reasoning capabilities using scientific tables. It enables the evaluation of reasoning-aware generation models.

- [**MRED: A Meta-Review Dataset for Structure-Controllable Text Generation**](https://arxiv.org/abs/2110.07474)</br>
  _Authors_: C. Shen, L. Cheng, R. Zhou, L. Bing, Y. You</br>
  _Description_: Presents MRED, a dataset aimed at enabling controllable text generation, particularly for summarizing and generating structured meta-reviews.

- [**ToTTo: A Controlled Table-to-Text Generation Dataset**](https://arxiv.org/abs/2004.14373)</br>
  _Authors_: A.P. Parikh, X. Wang, S. Gehrmann, M. Faruqui</br>
  _Description_: Proposes ToTTo, a dataset designed for controlled table-to-text generation tasks. It emphasizes generating text grounded on structured data.

- [**SciXGen: A Scientific Paper Dataset for Context-Aware Text Generation**](https://arxiv.org/abs/2110.10774)</br>
  _Authors_: H. Chen, H. Takamura, H. Nakayama</br>
  _Description_: Introduces SciXGen, a dataset that facilitates the development of models for context-aware scientific paper generation.

- [**DART: Open-Domain Structured Data Record to Text Generation**](https://arxiv.org/abs/2007.02871)</br>
  _Authors_: L. Nan, D. Radev, R. Zhang, A. Rau, A. Sivaprasad</br>
  _Description_: Presents DART, a dataset for transforming structured data records into coherent text, applicable in open-domain tasks.

- [**CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning**](https://arxiv.org/abs/1911.03705)</br>
  _Authors_: B.Y. Lin, W. Zhou, M. Shen, P. Zhou</br>
  _Description_: Introduces CommonGen, a dataset for testing constrained generative commonsense reasoning by generating coherent sentences grounded on given concepts.

- [**Evaluation of Text Generation: A Survey**](https://arxiv.org/abs/2006.14799)</br>
  _Authors_: A. Celikyilmaz, E. Clark, J. Gao</br>
  _Description_: Surveys various text generation datasets, models, and evaluation methods, offering insights into the current state and challenges of text generation.

### **10.2 Text Classification Datasets**

Text classification datasets help train models to categorize text into predefined labels. These datasets are used in applications like sentiment analysis, spam detection, and topic classification.

- [**NADA: New Arabic Dataset for Text Classification**](https://www.researchgate.net/profile/Souad-Larabi-Marie-Sainte/publication/327972319_NADA_New_Arabic_dataset_for_text_classification/links/5bd1607492851cabf265c549/NADA-New-Arabic-dataset-for-text-classification.pdf)</br>
  _Authors_: N. Alalyani, S. L. Marie-Sainte</br>
  _Description_: Introduces NADA, a structured and standardized dataset for Arabic text classification, addressing gaps in Arabic NLP datasets.

- [**Incremental Few-Shot Text Classification with Multi-Round New Classes: Formulation, Dataset and System**](https://arxiv.org/pdf/2104.11882)</br>
  _Authors_: C. Xia, W. Yin, Y. Feng, P. Yu</br>
  _Description_: Proposes a new benchmark dataset for incremental few-shot text classification, enabling evaluation of multi-round new class additions.

- [**Large-Scale Multi-Label Text Classification on EU Legislation**](https://arxiv.org/pdf/1906.02192)</br>
  _Authors_: I. Chalkidis, M. Fergadiotis, P. Malakasiotis</br>
  _Description_: Releases a new dataset of 57k legislative documents from EUR-LEX annotated with ∼4.3k labels for multi-label classification tasks.

- [**LSHTC: A Benchmark for Large-Scale Text Classification**](https://arxiv.org/pdf/1503.08581)</br>
  _Authors_: I. Partalas, A. Kosmopoulos, N. Baskiotis</br>
  _Description_: Introduces LSHTC, a benchmark dataset for hierarchical text classification, supporting tasks with hundreds of thousands of classes.

- [**Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach**](https://arxiv.org/pdf/1909.00161)</br>
  _Authors_: W. Yin, J. Hay, D. Roth</br>
  _Description_: Presents datasets tailored for zero-shot text classification with a standardized evaluation framework and entailment-based methods.

### **10.3 Named Entity Recognition Datasets**

Named Entity Recognition (NER) datasets are used for extracting named entities such as persons, locations, organizations, and dates from text. These datasets are crucial for tasks like information retrieval and knowledge extraction.

- [**Multimodal Named Entity Recognition for Short Social Media Posts**](https://arxiv.org/abs/1802.07862)</br>
  _Authors_: S. Moon, L. Neves, V. Carvalho</br>
  _Description_: Introduces a dataset for multimodal named entity recognition (MNER) in social media, leveraging both text and visual data for more robust recognition.

- [**MultiCoNER: A Large-Scale Multilingual Dataset for Complex Named Entity Recognition**](https://arxiv.org/abs/2208.14536)</br>
  _Authors_: S. Malmasi, A. Fang, B. Fetahu</br>
  _Description_: Presents MultiCoNER, a dataset designed to challenge NER models with fine-grained and complex entity recognition in a multilingual context.

- [**Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition**](https://aclanthology.org/W17-4418/)</br>
  _Authors_: L. Derczynski, E. Nichols, M. Van Erp</br>
  _Description_: Proposes a dataset for recognizing novel and emerging entities, emphasizing adaptability in dynamic domains like social media.

- [**Creating a Dataset for Named Entity Recognition in the Archaeology Domain**](https://aclanthology.org/2020.lrec-1.562/)</br>
  _Authors_: A. Brandsen, S. Verberne, M. Wansleeben</br>
  _Description_: Develops a domain-specific NER dataset tailored to archaeological texts, annotated with six custom entity types.

- [**NNE: A Dataset for Nested Named Entity Recognition in English Newswire**](https://aclanthology.org/P19-1510/)</br>
  _Authors_: N. Ringland, X. Dai, B. Hachey, S. Karimi, C. Paris</br>
  _Description_: Introduces NNE, a large-scale dataset for nested entity recognition, pushing models to handle hierarchical structures in newswire data.

- [**CLUENER2020: Fine-Grained Named Entity Recognition Dataset and Benchmark for Chinese**](https://arxiv.org/abs/2001.04351)</br>
  _Authors_: L. Xu, Q. Dong, Y. Liao, C. Yu</br>
  _Description_: Presents CLUENER2020, a challenging dataset for fine-grained NER in Chinese, incorporating new entity types and samples.

- [**Crosslingual Named Entity Recognition for Clinical De-Identification Applied to a COVID-19 Italian Dataset**](https://www.sciencedirect.com/science/article/pii/S1568494620307171)</br>
  _Authors_: R. Catelli, F. Gargiulo, V. Casola, G. De Pietro</br>
  _Description_: Creates a new dataset of Italian COVID-19 clinical records for cross-lingual NER, focusing on de-identification and anonymization.

### **10.4 Question Answering Datasets**

Question Answering (QA) datasets enable models to generate answers based on a given question and context. These datasets are widely used in search engines, virtual assistants, and automated customer support systems.

- [**WikiQA: A Challenge Dataset for Open-Domain Question Answering**](https://aclanthology.org/D15-1237.pdf)</br>
  _Authors_: Y. Yang, W. Yih, C. Meek</br>
  _Description_: Introduces WikiQA, a dataset for open-domain question answering, constructed from natural and realistic queries on Wikipedia.

- [**GQA: A New Dataset for Compositional Question Answering Over Real-World Images**](https://www.thetalkingmachines.com/sites/default/files/2019-03/1902.09506-compressed.pdf)</br>
  _Authors_: D.A. Hudson, C.D. Manning</br>
  _Description_: Proposes GQA, a dataset for visual reasoning and compositional question answering, designed to address key shortcomings of visual QA datasets.

- [**HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering**](https://arxiv.org/abs/1809.09600)</br>
  _Authors_: Z. Yang, P. Qi, S. Zhang, Y. Bengio, W.W. Cohen</br>
  _Description_: Introduces HotpotQA, a dataset emphasizing diverse and explainable multi-hop reasoning tasks using Wikipedia as its knowledge base.

- [**ToolQA: A Dataset for Question Answering with External Tools**](https://proceedings.neurips.cc/paper_files/paper/2023/file/9cb2a7495900f8b602cb10159246a016-Paper-Datasets_and_Benchmarks.pdf)</br>
  _Authors_: Y. Zhuang, Y. Yu, K. Wang, H. Sun</br>
  _Description_: Proposes ToolQA, a dataset for exploring the integration of external tools with question answering systems.

- [**QASC: A Dataset for Question Answering via Sentence Composition**](https://ojs.aaai.org/index.php/AAAI/article/view/6319)</br>
  _Authors_: T. Khot, P. Clark, M. Guerquin, P. Jansen</br>
  _Description_: Introduces QASC, a dataset focusing on multi-hop reasoning through sentence composition to answer multiple-choice questions.

- [**What Do Models Learn from Question Answering Datasets?**](https://arxiv.org/abs/2004.03490)</br>
  _Authors_: P. Sen, A. Saffari</br>
  _Description_: Explores generalizability across question answering datasets and highlights challenges with impossible questions in dataset design.

- [**QA Dataset Explosion: A Taxonomy of NLP Resources for Question Answering**](https://arxiv.org/pdf/2107.12708)</br>
  _Authors_: A. Rogers, M. Gardner, I. Augenstein</br>
  _Description_: Analyzes the proliferation of question answering datasets, providing a taxonomy of more than 80 resources in QA and reading comprehension.

### **10.5 Fill Mask Datasets**

Fill Mask datasets are used for training masked language models (MLMs) where a model learns to predict missing words in a given sentence. These datasets help improve contextualized word representations.

- [**The Fill-Mask Association Test (FMAT): Measuring Propositions in Natural Language**](https://psychbruce.github.io/paper/Bao_Accepted_JPSP_FMAT_Manuscript.pdf)</br>
  _Authors_: L. Lin, B. Wang, X. Wang, Z.X. Wang, A. Wiśniowski</br>
  _Description_: Introduces FMAT, a dataset designed to measure semantic probabilities in natural language using fill-mask tasks for evaluating language models.

- [**Performance Implications of Using Unrepresentative Corpora in Arabic NLP**](https://aclanthology.org/2023.arabicnlp-1.19/)</br>
  _Authors_: S. Alshahrani, N. Alshahrani, S. Dey</br>
  _Description_: Creates a dataset for evaluating fill-mask tasks in Arabic, addressing the challenges posed by unrepresentative corpora in language modeling.

- [**Automated Distractor Generation for Fill-in-the-Blank Items Using a Prompt-Based Learning Approach**](https://www.psychologie-aktuell.com/fileadmin/Redaktion/Journale/ptam_2023-1/PTAM__1-2023_3_kor.pdf)</br>
  _Authors_: J. Zu, I. Choi, J. Hao</br>
  _Description_: Proposes a new dataset for fill-in-the-blank tasks, leveraging prompt-based learning to generate distractors automatically.

- [**DarkBERT: A Language Model for the Dark Side of the Internet**](https://arxiv.org/pdf/2305.08596)</br>
  _Authors_: Y. Jin, E. Jang, J. Cui, J.W. Chung, Y. Lee</br>
  _Description_: Presents a dataset tailored for cybersecurity tasks, with evaluations on fill-mask and synonym inference capabilities.

- [**We Understand Elliptical Sentences, and Language Models Should Too**](https://aclanthology.org/2023.acl-long.188/)</br>
  _Authors_: D. Testa, E. Chersoni, A. Lenci</br>
  _Description_: Creates a dataset for studying ellipsis and its interaction with thematic fit, focusing on fill-mask tasks to predict missing verbs.

- [**Iterative Mask Filling: An Effective Text Augmentation Method Using Masked Language Modeling**](https://link.springer.com/chapter/10.1007/978-3-031-50920-9_35)</br>
  _Authors_: H.T. Kesgin, M.F. Amasyali</br>
  _Description_: Proposes a dataset and methodology for iterative mask filling, designed to augment text effectively through masked language modeling.

- [**Efficient and Thorough Anonymizing of Dutch Electronic Health Records**](https://aclanthology.org/2022.lrec-1.118/)</br>
  _Authors_: S. Verkijk, P. Vossen</br>
  _Description_: Develops a dataset for anonymizing Dutch electronic health records using fill-mask tasks as part of the de-identification process.

### **10.6 Machine Translation Datasets**

Machine translation datasets provide parallel corpora for training models to translate text between different languages. These datasets are fundamental in developing multilingual NLP systems.

- [**The Fill-Mask Association Test (FMAT): Measuring Propositions in Natural Language**](https://psychbruce.github.io/paper/Bao_Accepted_JPSP_FMAT_Manuscript.pdf)</br>
  _Authors_: L. Lin, B. Wang, X. Wang, Z.X. Wang, A. Wiśniowski</br>
  _Description_: Introduces FMAT, a dataset designed to measure semantic probabilities in natural language using fill-mask tasks for evaluating language models.

- [**Performance Implications of Using Unrepresentative Corpora in Arabic NLP**](https://aclanthology.org/2023.arabicnlp-1.19/)</br>
  _Authors_: S. Alshahrani, N. Alshahrani, S. Dey</br>
  _Description_: Creates a dataset for evaluating fill-mask tasks in Arabic, addressing the challenges posed by unrepresentative corpora in language modeling.

- [**Automated Distractor Generation for Fill-in-the-Blank Items Using a Prompt-Based Learning Approach**](https://www.psychologie-aktuell.com/fileadmin/Redaktion/Journale/ptam_2023-1/PTAM__1-2023_3_kor.pdf)</br>
  _Authors_: J. Zu, I. Choi, J. Hao</br>
  _Description_: Proposes a new dataset for fill-in-the-blank tasks, leveraging prompt-based learning to generate distractors automatically.

- [**DarkBERT: A Language Model for the Dark Side of the Internet**](https://arxiv.org/pdf/2305.08596)</br>
  _Authors_: Y. Jin, E. Jang, J. Cui, J.W. Chung, Y. Lee</br>
  _Description_: Presents a dataset tailored for cybersecurity tasks, with evaluations on fill-mask and synonym inference capabilities.

- [**We Understand Elliptical Sentences, and Language Models Should Too**](https://aclanthology.org/2023.acl-long.188/)</br>
  _Authors_: D. Testa, E. Chersoni, A. Lenci</br>
  _Description_: Creates a dataset for studying ellipsis and its interaction with thematic fit, focusing on fill-mask tasks to predict missing verbs.

- [**Iterative Mask Filling: An Effective Text Augmentation Method Using Masked Language Modeling**](https://link.springer.com/chapter/10.1007/978-3-031-50920-9_35)</br>
  _Authors_: H.T. Kesgin, M.F. Amasyali</br>
  _Description_: Proposes a dataset and methodology for iterative mask filling, designed to augment text effectively through masked language modeling.

- [**Efficient and Thorough Anonymizing of Dutch Electronic Health Records**](https://aclanthology.org/2022.lrec-1.118/)</br>
  _Authors_: S. Verkijk, P. Vossen</br>
  _Description_: Develops a dataset for anonymizing Dutch electronic health records using fill-mask tasks as part of the de-identification process.

## **11. NLP in Vietnamese**

Vietnamese NLP presents unique challenges due to the language's lack of word boundaries, tonal nature, and rich morphology. This section provides a collection of papers, tools, and datasets specifically tailored for Vietnamese NLP research and applications.

### **11.1 Vietnamese Text Preprocessing**

Vietnamese text preprocessing involves tasks such as tokenization, stopword removal, and diacritic normalization. Due to the lack of explicit word boundaries, word segmentation is a critical preprocessing step in Vietnamese NLP.

- [**Vietnamese Text Classification with Textrank and Jaccard Similarity Coefficient**](https://www.academia.edu/download/99306430/ASTESJ_050644.pdf)</br>
  _Authors_: HT Huynh, N Duong-Trung, DQ Truong</br>
  _Description_: Proposes a preprocessing pipeline for Vietnamese text classification using Textrank for keyword extraction and Jaccard similarity for feature selection.

- [**Vietnamese Short Text Classification via Distributed Computation**](https://www.researchgate.net/publication/353608862_Vietnamese_Short_Text_Classification_via_Distributed_Computation)</br>
  _Authors_: HX Huynh, LX Dang, N Duong-Trung</br>
  _Description_: Explores preprocessing techniques for Vietnamese short text classification, focusing on distributed computation approaches.

- [**DaNangNLP Toolkit for Vietnamese Text Preprocessing and Word Segmentation**](https://elib.vku.udn.vn/bitstream/123456789/4041/1/P26.296-307.pdf)</br>
  _Authors_: KD Nguyen, TT Nguyen, DB Nguyen</br>
  _Description_: Develops a comprehensive toolkit for Vietnamese text preprocessing, including tokenization, word segmentation, and normalization.

- [**Feature Extraction Using Neural Networks for Vietnamese Text Classification**](https://ieeexplore.ieee.org/abstract/document/9418674/)</br>
  _Authors_: HH Kha</br>
  _Description_: Proposes feature extraction techniques for Vietnamese text preprocessing using neural networks to enhance classification accuracy.

- [**ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing**](https://arxiv.org/abs/2310.11166)</br>
  _Authors_: QN Nguyen, TC Phan, DV Nguyen</br>
  _Description_: Introduces ViSoBERT, a pre-trained model tailored for Vietnamese social media text, focusing on robust preprocessing pipelines.

- [**SVSD: A Comprehensive Framework for Vietnamese Sentiment Analysis**](https://link.springer.com/chapter/10.1007/978-981-96-0434-0_26)</br>
  _Authors_: LT Nhi, DHA Vu, VDT Phong</br>
  _Description_: Presents preprocessing steps and sentiment analysis methods for Vietnamese text to ensure data uniformity and effective modeling.

- [**An Empirical Study on POS Tagging for Vietnamese Social Media Text**](https://www.sciencedirect.com/science/article/pii/S088523081730027X)</br>
  _Authors_: NX Bach, ND Linh, TM Phuong</br>
  _Description_: Focuses on part-of-speech tagging as a preprocessing task for Vietnamese social media text, creating a dataset for this task.

### **11.2 Vietnamese Word Representations**

Word embeddings and contextualized word representations trained specifically for Vietnamese text improve NLP performance. This includes models like Word2Vec, FastText, and transformer-based embeddings such as PhoBERT.

- [**Construction of a VerbNet Style Lexicon for Vietnamese**](https://aclanthology.org/2020.paclic-1.10.pdf)</br>
  _Authors_: H.M. Linh, N.T.M. Huyen</br>
  _Description_: Develops a lexicon for Vietnamese verbs using word2vec representations on a large corpus, enabling applications in parsing and semantic tasks.

- [**Comparing Different Criteria for Vietnamese Word Segmentation**](https://aclanthology.org/W12-5005.pdf)</br>
  _Authors_: Q. Nguyen, N.L.T. Nguyen, Y. Miyao</br>
  _Description_: Explores criteria for Vietnamese word segmentation and its impact on the quality of word representations in downstream tasks.

- [**Improving Vietnamese Dependency Parsing Using Distributed Word Representations**](https://www.researchgate.net/publication/282968208)</br>
  _Authors_: C. Vu-Manh, A.T. Luong, P. Le-Hong</br>
  _Description_: Investigates how distributed word embeddings improve dependency parsing for Vietnamese, achieving significant accuracy improvements.

- [**A Study of Word Representation in Vietnamese Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/9538456/)</br>
  _Authors_: H.Q. Nguyen, L. Vu, Q.U. Nguyen</br>
  _Description_: Evaluates various word representation methods for sentiment analysis, focusing on Vietnamese corpora and sentiment tasks.

- [**Leveraging Semantic Representations Combined with Contextual Word Representations for Vietnamese Textual Entailment**](https://ieeexplore.ieee.org/abstract/document/10013423/)</br>
  _Authors_: Q.L. Duong, D.V. Nguyen</br>
  _Description_: Combines semantic and contextual representations to improve performance on Vietnamese textual entailment tasks.

- [**Vietnamese Document Representation and Classification**](https://link.springer.com/chapter/10.1007/978-3-642-10439-8_58)</br>
  _Authors_: G.S. Nguyen, X. Gao, P. Andreae</br>
  _Description_: Proposes document-level representation techniques for Vietnamese, including bag-of-words and semantic embeddings.

- [**Deep Neural Networks Algorithm for Vietnamese Word Segmentation**](https://onlinelibrary.wiley.com/doi/abs/10.1155/2022/8187680)</br>
  _Authors_: K. Zheng, W. Zheng</br>
  _Description_: Presents a deep neural network-based approach for Vietnamese word segmentation, leveraging contextualized embeddings for superior accuracy.

### **11.3 Vietnamese Named Entity Recognition (NER)**

Named Entity Recognition (NER) identifies entities such as names, organizations, and locations within Vietnamese text. Challenges include handling ambiguous entity boundaries and diacritic variations.

- [**Named Entity Recognition in Vietnamese Documents**](https://www.academia.edu/download/83622008/Named_entity_recognition_in_vietnamese_d20220409-17153-vnnoij.pdf)</br>
  _Authors_: QT Tran, TXT Pham, QH Ngo, D Dinh</br>
  _Description_: Explores techniques for recognizing named entities in Vietnamese documents with a focus on extracting relations and tracking entities across texts.

- [**A Feature-Rich Vietnamese Named Entity Recognition Model**](https://www.scielo.org.mx/scielo.php?pid=S1405-55462022000301323&script=sci_arttext&tlng=en)</br>
  _Authors_: PQ Nhat Minh</br>
  _Description_: Presents a feature-rich NER model for Vietnamese that achieves state-of-the-art accuracy by combining multiple NLP toolkits and advanced chunking methods.

- [**On the Vietnamese Name Entity Recognition: A Deep Learning Method Approach**](https://ieeexplore.ieee.org/abstract/document/9140754/)</br>
  _Authors_: NC Lê, NY Nguyen, AD Trinh</br>
  _Description_: Investigates the application of deep learning methods to Vietnamese NER, demonstrating state-of-the-art performance using contextual embeddings.

- [**The Importance of Automatic Syntactic Features in Vietnamese Named Entity Recognition**](https://arxiv.org/abs/1705.10610)</br>
  _Authors_: TH Pham, P Le-Hong</br>
  _Description_: Highlights the role of syntactic features in improving Vietnamese NER, utilizing an automatic feature extraction approach.

- [**Vietnamese Named Entity Recognition on Medical Topics**](https://www.researchgate.net/publication/362082936_Vietnamese_Named_Entity_Recognition_on_Medical_Topic)</br>
  _Authors_: DP Van, DN Tien, TTT Minh, TD Minh</br>
  _Description_: Proposes a new NER dataset for Vietnamese medical texts, including newly defined entity types and extensive annotations.

- [**COVID-19 Named Entity Recognition for Vietnamese**](https://arxiv.org/abs/2104.03879)</br>
  _Authors_: TH Truong, MH Dao, DQ Nguyen</br>
  _Description_: Develops a COVID-19 domain-specific dataset for Vietnamese NER, incorporating novel entity types and robust annotations.

- [**ViMedNER: A Medical Named Entity Recognition Dataset for Vietnamese**](https://publications.eai.eu/index.php/inis/article/download/5221/3275)</br>
  _Authors_: P Van Duong, TD Trinh, MT Nguyen</br>
  _Description_: Introduces ViMedNER, a dataset focused on medical entity recognition, specifically tailored for Vietnamese texts.

### **11.4 Vietnamese Part-of-Speech Tagging**

Part-of-Speech (POS) tagging in Vietnamese requires models to correctly classify words into grammatical categories despite the language’s complex morphology and word segmentation issues.

- [**A Semi-Supervised Learning Method for Vietnamese Part-of-Speech Tagging**](https://ieeexplore.ieee.org/abstract/document/5632134/)</br>
  _Authors_: BN Xuan, CN Viet, MPQ Nhat</br>
  _Description_: Proposes a semi-supervised learning approach for Vietnamese POS tagging, combining perceptron-based and tagging-style models.

- [**Comparative Study of Vietnamese Part-of-Speech Tagging Tools**](https://ieeexplore.ieee.org/abstract/document/9232564/)</br>
  _Authors_: LD Quach, D Do Thanh, DC Tran</br>
  _Description_: Presents a comparative analysis of existing Vietnamese POS tagging tools and evaluates their accuracy and efficiency.

- [**An Empirical Study of Maximum Entropy Approach for Part-of-Speech Tagging of Vietnamese Texts**](https://inria.hal.science/inria-00526139/)</br>
  _Authors_: P Le-Hong, A Roussanaly, TMH Nguyen</br>
  _Description_: Explores the application of the maximum entropy model for Vietnamese POS tagging, leveraging a wide range of linguistic features.

- [**PhoNLP: A Joint Multi-Task Learning Model for Vietnamese POS Tagging, NER, and Dependency Parsing**](https://arxiv.org/abs/2101.01476)</br>
  _Authors_: LT Nguyen, DQ Nguyen</br>
  _Description_: Introduces PhoNLP, a joint model for POS tagging, named entity recognition, and dependency parsing, demonstrating state-of-the-art performance.

- [**An Empirical Study on POS Tagging for Vietnamese Social Media Text**](https://www.sciencedirect.com/science/article/pii/S088523081730027X)</br>
  _Authors_: NX Bach, ND Linh, TM Phuong</br>
  _Description_: Focuses on adapting POS tagging to handle the unique challenges of Vietnamese social media text.

- [**A Hybrid Approach to Vietnamese Word Segmentation Using POS Tags**](https://ieeexplore.ieee.org/abstract/document/5361713/)</br>
  _Authors_: GB Tran, SB Pham</br>
  _Description_: Develops a hybrid approach integrating POS tagging to improve Vietnamese word segmentation techniques.

- [**Dual Decomposition for Vietnamese Part-of-Speech Tagging**](https://www.sciencedirect.com/science/article/pii/S1877050913008818)</br>
  _Authors_: NX Bach, K Hiraishi, N Le Minh, A Shimazu</br>
  _Description_: Proposes a dual decomposition method for Vietnamese POS tagging, addressing limitations in existing models.

### **11.5 Vietnamese Syntax and Parsing**

Vietnamese dependency parsing and constituency parsing help analyze sentence structures, enabling downstream applications like machine translation and question answering.

- [**Prosodic Phrasing Modeling for Vietnamese TTS Using Syntactic Information**](https://www.researchgate.net/publication/367864197_Prosodic_phrasing_modeling_for_vietnamese_TTS_using_syntactic_information)</br>
  _Authors_: NTT Trang, A Rilliard, T Do Dat</br>
  _Description_: Explores the interface between syntax and prosody in Vietnamese text-to-speech (TTS) systems, leveraging syntactic information to improve phrasing.

- [**Semantic Parsing for Vietnamese: A Cross-Lingual Approach**](https://cs.brynmawr.edu/Courses/cs399/spring2023/Final/TessaPham.pdf)</br>
  _Authors_: T Pham</br>
  _Description_: Presents a cross-lingual approach to semantic parsing for Vietnamese, focusing on syntactic and semantic challenges.

- [**Vietnamese Parsing Applying the PCFG Model**](https://users.soict.hust.edu.vn/vietha/papers/PCFG%20Parsing.pdf)</br>
  _Authors_: HA Viet, DTP Thu, HQ Thang</br>
  _Description_: Investigates the use of probabilistic context-free grammar (PCFG) for Vietnamese syntax parsing, enhancing parsing accuracy.

- [**Building a Treebank for Vietnamese Syntactic Parsing**](https://ir.soken.ac.jp/?action=repository_uri&item_id=5684&file_id=19&file_no=2)</br>
  _Authors_: NT Quy</br>
  _Description_: Develops a Vietnamese treebank and evaluates different parsing methods, identifying sources of parsing errors.

- [**Semantic Parsing of Simple Sentences in Unification-Based Vietnamese Grammar**](https://www.academia.edu/download/35555656/07.pdf)</br>
  _Authors_: DT Nguyen, KD Nguyen, HT Le</br>
  _Description_: Explores unification-based grammar for semantic parsing of simple Vietnamese sentences, emphasizing taxonomy and grammar development.

- [**An Experimental Study on Constituency Parsing for Vietnamese**](https://link.springer.com/chapter/10.1007/978-981-15-6168-9_30)</br>
  _Authors_: L Nguyen-Thi, P Le-Hong</br>
  _Description_: Analyzes constituency parsing for Vietnamese using syntax-annotated corpora, presenting empirical results and model performance.

- [**Using Syntax and Shallow Semantic Analysis for Vietnamese Question Generation**](https://koreascience.kr/article/JAKO202331453098308.page)</br>
  _Authors_: P Tran, DK Nguyen, T Tran, B Vo</br>
  _Description_: Applies syntax and shallow semantic analysis to Vietnamese question generation, addressing limitations in existing models.

### **11.6 Machine Translation for Vietnamese**

Machine translation between Vietnamese and other languages (e.g., English, French, Chinese) is an active research area. Transformer-based models like MarianMT and multilingual BERT-based models improve translation quality.

- [**ViBidirectionMT-Eval: Machine Translation for Vietnamese-Chinese and Vietnamese-Lao language pairs**](https://arxiv.org/abs/2501.08621)</br>
  _Authors_: HV Tran, MQ Nguyen, VV Nguyen</br>
  _Description_: This study evaluates bidirectional machine translation between Vietnamese-Chinese and Vietnamese-Lao, focusing on fluency and accuracy."

- [**Are LLMs Good for Low-resource Vietnamese and Other Translations?**](https://www.researchsquare.com/article/rs-5355866/latest)</br>
  _Authors_: VV Nguyen, H Nguyen-Tien, P Nguyen-Ngoc</br>
  _Description_: Investigates the performance of large language models (LLMs) in low-resource translation tasks, including Vietnamese."

- [**Handling Imbalanced Resources and Loanwords in Vietnamese-Bahnaric Neural Machine Translation**](https://www.inderscienceonline.com/doi/abs/10.1504/IJIIDS.2024.141776)</br>
  _Authors_: LNH Bui, HTP Nguyen, MK Le</br>
  _Description_: Focuses on neural machine translation for the Vietnamese-Bahnaric language pair, tackling issues of imbalanced data and loanwords."

- [**Constructing a Chinese-Vietnamese Bilingual Corpus from Subtitle Websites**](https://www.inderscienceonline.com/doi/abs/10.1504/IJIIDS.2024.141748)</br>
  _Authors_: PN Nguyen, P Tran</br>
  _Description_: Explores using subtitle data to build a high-quality Vietnamese-Chinese parallel corpus."

- [**Exploring Low-Resource Machine Translation: Case Study of Lao-Vietnamese Translation**](https://ieeexplore.ieee.org/abstract/document/10660932/)</br>
  _Authors_: QD Tran</br>
  _Description_: Develops a machine translation system for the low-resource Vietnamese-Lao language pair."

- [**Neural Network Translations for Building SentiWordNets**](https://link.springer.com/article/10.1007/s10844-024-00911-7)</br>
  _Authors_: KN Lam, TP Le, KC Ngu, KT Le, PM Le</br>
  _Description_: Uses machine translation to create a Vietnamese version of the SentiWordNet lexical resource."

- [**Evaluating the Feasibility of Machine Translation for Patient Education in Vietnamese**](https://www.sciencedirect.com/science/article/pii/S0738399124004270)</br>
  _Authors_: M Ugas, MA Calamia, J Tan, B Umakanthan</br>
  _Description_: Assesses Google Translate for translating patient education materials into Vietnamese."

- [**Improving Chinese-Vietnamese Neural Machine Translation with Irrelevant Word Detection**](https://ieeexplore.ieee.org/abstract/document/10800263/)</br>
  _Authors_: T Wang, Z Yu, W Yu, W Sun</br>
  _Description_: Introduces a method to filter irrelevant words to improve Vietnamese-Chinese machine translation."

### **11.7 Vietnamese Question Answering**

Question Answering (QA) systems in Vietnamese involve answering questions based on structured or unstructured text. QA models require high-quality annotated datasets for accurate responses.

- [**Building a Website to Sell Electronic Devices Store Integrated with Chatbot AI and VNPay Payment Gateway**](https://elib.vku.udn.vn/bitstream/123456789/4572/1/20SE2-20IT1023.%20Nguyen%20Thanh%20Tung-20SE6-20IT721.%20Nguyen%20Van%20Nhat.pdf)</br>
  _Authors_: TT Nguyen, VN Nguyen</br>
  _Description_: This study explores the integration of AI chatbots in e-commerce, specifically within Vietnamese electronic stores using VNPay."

- [**Top 2 at ALQAC 2024: Large Language Models (LLMs) for Legal Question Answering**](https://www.worldscientific.com/doi/abs/10.1142/S2717554524500103)</br>
  _Authors_: HQ Pham, Q Van Nguyen, DQ Tran</br>
  _Description_: Analyzes the use of large language models (LLMs) for legal question answering in Vietnamese law."

- [**Critical Discourse Analysis of Judicial Conversations in Vietnam: A Case Study**](https://ideas.repec.org/a/jfr/wjel11/v15y2025i1p362.html)</br>
  _Authors_: PT Ly</br>
  _Description_: Examines the structure and discourse of judicial question-answer interactions in Vietnamese courts."

- [**Vietnamese Young People and the Reactive Public Sphere**](https://link.springer.com/chapter/10.1007/978-981-97-8955-9_6)</br>
  _Authors_: VT Le, TM Ly-Le, L Ha</br>
  _Description_: Investigates how young Vietnamese individuals engage in public discourse and answer political questions in online spaces."

- [**Four Important Characteristics of Women in Confucianism and Its Contribution to the Implementation o**](Gender Equality in Vietnam",https://ejournals.epublishing.ekt.gr/index.php/Conatus/article/view/35243)</br>
  _Authors_: D Van Vo</br>
  _Description_: Discusses how Confucianism has shaped gender roles and question-answer dynamics in Vietnamese society."

- [**Man in a Hurry: Murray MacLehose and Colonial Autonomy in Hong Kong**](https://www.cambridge.org/core/journals/china-quarterly/article/man-in-a-hurry-murray-maclehose-and-colonial-autonomy-in-hong-kong-ray-yep-hong-kong-hong-kong-university-press-2024-xi-202-pp-hk29500-hbk-isbn-9898888842926/0C3546EA2CEABEFADB08F0CD56DC0EF1)</br>
  _Authors_: P Roberts</br>
  _Description_: Explores how Vietnamese refugees' legal and political questions were addressed in colonial Hong Kong."

- [**Integrating Theatrical Arts into Storytelling Instruction in Primary Education**](https://iejee.com/index.php/IEJEE/article/view/2240)</br>
  _Authors_: QV Tran, YN Tran</br>
  _Description_: Examines how question-answer techniques in storytelling can be improved with theatrical methods in Vietnamese schools."

- [**Buddhism: A Journey through History**](https://books.google.com/books?hl=en&lr=&id=c1g5EQAAQBAJ&oi=fnd&pg=PP1&dq=Vietnamese+Question+Answering&ots=l7wQamigsA&sig=vEGvDxqhh5dz0rL7cX5Xjppbi_A)</br>
  _Authors_: DS Lopez</br>
  _Description_: Explores how Buddhism has historically answered philosophical and religious questions in Vietnam."

### **11.8 Vietnamese Text Summarization**

Text summarization generates concise and informative summaries from long Vietnamese documents. Extractive and abstractive summarization techniques are commonly used for this task.

- [**Vietnamese Online Newspapers Summarization Using Pre-Trained Model**](https://apni.ru/uploads/ai_2_1_2024.pdf#page=10)</br>
  _Authors_: T Le Ngoc</br>
  _Description_: Presents a model for summarizing Vietnamese online newspapers using pre-trained deep learning techniques."

- [**Graph-based and Generative Approaches to Multi-Document Summarization**](https://vjs.ac.vn/index.php/jcc/article/view/18353)</br>
  _Authors_: TD Thanh, TM Nguyen, TB Nguyen, HT Nguyen</br>
  _Description_: Introduces a hybrid approach combining graph-based and generative methods for Vietnamese multi-document summarization."

- [**THASUM: Transformer for High-Performance Abstractive Summarizing Vietnamese Large-scale Dataset**](https://elib.vku.udn.vn/handle/123456789/4019)</br>
  _Authors_: TH Nguyen, TN Do</br>
  _Description_: Develops a transformer-based abstractive summarization model trained on a large-scale Vietnamese dataset."

- [**Pre-Training Clustering Models to Summarize Vietnamese Texts**](https://www.worldscientific.com/doi/abs/10.1142/S2196888824500118)</br>
  _Authors_: TH Nguyen, TN Do</br>
  _Description_: Proposes a clustering-based pre-training approach for single-document extractive summarization in Vietnamese."

- [**Vietnamese Online Newspapers Summarization Using LexRank**](https://elibrary.ru/item.asp?id=60052519)</br>
  _Authors_: LEN THANG, LEQ MINH</br>
  _Description_: Applies the LexRank algorithm for Vietnamese news summarization using graph-based sentence ranking."

- [**Feature-Based Unsupervised Method for Salient Sentence Ranking in Text Summarization Task**](https://dl.acm.org/doi/abs/10.1145/3654522.3654556)</br>
  _Authors_: MP Nguyen, TA Le</br>
  _Description_: Develops an unsupervised sentence ranking model for Vietnamese text summarization."

- [**Paraphrasing with Large Language Models**](https://elib.vku.udn.vn/handle/123456789/4023)</br>
  _Authors_: CT Nguyen, DHP Pham, CT Dang, TH Le</br>
  _Description_: Explores the use of large language models for Vietnamese text paraphrasing and summarization."

- [**Resource-Efficient Vietnamese Text Summarization**](https://link.springer.com/chapter/10.1007/978-981-96-0437-1_22)</br>
  _Authors_: HD Nguyen Pham, DT Nguyen</br>
  _Description_: Enhances the efficiency of Vietnamese text summarization using data filtering and low-memory deep learning techniques."

### **11.9 Resources for Vietnamese NLP**

A collection of open-source tools, frameworks, and datasets for Vietnamese NLP, including word segmentation tools, language models, and benchmark datasets.

- [**Automatically Generating a Dataset for Natural Language Inference Systems from a Knowledge Graph**](https://link.springer.com/chapter/10.1007/978-981-97-9616-8_21)</br>
  _Authors_: DV Vo, P Do</br>
  _Description_: Presents a dataset for Vietnamese Natural Language Inference (NLI) using a knowledge graph, contributing to NLP research and model evaluation."

- [**Neural Network Translations for Building SentiWordNets**](https://link.springer.com/article/10.1007/s10844-024-00911-7)</br>
  _Authors_: KN Lam, TP Le, KC Ngu, KT Le, PM Le</br>
  _Description_: Explores neural network-based translation for creating Vietnamese SentiWordNet, enhancing sentiment analysis resources."

- [**Updated Activities on Resources Development for Vietnamese Speech and NLP**](https://ieeexplore.ieee.org/abstract/document/10800426/)</br>
  _Authors_: LC Mai</br>
  _Description_: Reviews recent developments in Vietnamese NLP and speech resources, including government initiatives and industry collaborations."

### **11.10 Challenges in Vietnamese NLP**

Discusses the key challenges in Vietnamese NLP, such as handling tonal variations, segmentation difficulties, data scarcity, and the need for high-quality annotated datasets.

- [**Evaluating the Effectiveness of Commonly Used Sentiment Analysis Models for the Second Indochina War**](https://www.preprints.org/manuscript/202412.1670/download/final_file)</br>
  _Authors_: A Chakraborty</br>
  _Description_: Examines challenges in applying sentiment analysis models to Vietnamese historical texts, highlighting limitations in existing NLP approaches."

- [**Machine Learning Approach for Suicide and Depression Identification with Corrected Unsupervised Labels**](https://easychair.org/publications/preprint/tl5Q/download)</br>
  _Authors_: M Badki</br>
  _Description_: Discusses the challenges of identifying mental health-related text in Vietnamese using machine learning models with unsupervised labels."

- [**Building A Job Portal Website Integrating AI Technology**](https://elib.vku.udn.vn/bitstream/123456789/4568/1/20SE2-20IT804.%20Nguyen%20Phuoc%20Thinh-20SE6-20IT425.%20Nguyen%20Thi%20Hong%20Hanh.pdf)</br>
  _Authors_: PT Nguyen, THH Nguyen</br>
  _Description_: Explores NLP challenges in building AI-powered job search platforms for Vietnamese users."

- [**ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization**](https://arxiv.org/abs/2501.07020)</br>
  _Authors_: ATH Nguyen, DH Nguyen, K Van Nguyen</br>
  _Description_: Addresses lexical normalization issues in Vietnamese social media text processing."

- [**A Weakly Supervised Data Labeling Framework for Machine Lexical Normalization in Vietnamese Social Media**](https://arxiv.org/abs/2409.20467)</br>
  _Authors_: DH Nguyen, ATH Nguyen, K Van Nguyen</br>
  _Description_: Proposes a weakly supervised labeling approach to tackle low-resource challenges in Vietnamese NLP."

- [**VNLegalEase: A Vietnamese Legal Query Chatbot**](https://link.springer.com/chapter/10.1007/978-981-97-9616-8_23)</br>
  _Authors_: PTX Hien, NTT Vy, HD Ngo</br>
  _Description_: Discusses NLP difficulties in legal document understanding and chatbot development for Vietnamese law."

- [**Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges**](https://aclanthology.org/2024.emnlp-main.426/)</br>
  _Authors_: N Dinh, T Dang, LT Nguyen</br>
  _Description_: Investigates dialectal variation in Vietnamese and its impact on NLP tasks and model performance."

- [**Contextual Emotional Transformer-Based Model for Comment Analysis in Mental Health Case Prediction**](https://gala.gre.ac.uk/id/eprint/49329/)</br>
  _Authors_: AOJ Ibitoye, OO Oladosu</br>
  _Description_: Explores the challenges of contextual emotion detection in Vietnamese NLP for mental health prediction."

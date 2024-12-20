
# AI Reading List

<img
src="docs/posters/llama_apple_tree.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

# Getting Started with Hugging Face

Welcome to the exciting world of Hugging Face! This chapter is designed to introduce you to the powerful tools and libraries that have revolutionized the field of Natural Language Processing (NLP) and machine learning. Whether you're a complete beginner or an experienced developer looking to expand your AI toolkit, you'll find valuable insights and practical knowledge in the following articles.

## What You'll Learn

1. **Hugging Face Transformers Demystified**: We'll start with a beginner-friendly introduction to Hugging Face Transformers. This article aims to break down complex concepts into digestible pieces, helping you understand the core principles without getting lost in technical jargon. By the end, you'll have a solid grasp of what Transformers are and how they work, enabling you to collaborate more effectively with machine learning practitioners.

2. **Uncovering Hugging Face's Hidden Gems**: Finally, we'll dive into some of the most underrated tools on the Hugging Face platform. This guide will introduce you to powerful yet often overlooked resources like ZeroGPU, Gradio API, and Nomic Atlas. You'll discover how these tools can enhance your AI projects, from optimizing GPU usage to creating interactive demos and visualizing complex datasets.

By the end of this chapter, you'll have a comprehensive understanding of the Hugging Face ecosystem and the confidence to start exploring its vast potential. Whether you're interested in building language models, creating AI-powered applications, or simply staying up-to-date with the latest in NLP technology, these articles will provide you with a solid foundation to build upon.

Let's embark on this journey into the heart of modern AI development with Hugging Face!

## [Total noob’s intro to Hugging Face Transformers](https://huggingface.co/blog/noob_intro_transformers)

Our goal is to demystify what Hugging Face Transformers is and how it works, not to turn you into a machine learning practitioner, but to enable better understanding of and collaboration with those who are.

## [The 5 Most Under-Rated Tools on Hugging Face](https://huggingface.co/blog/unsung-heroes)

Supercharge your AI projects with these hidden gems on Hugging Face! Learn how to use tools like ZeroGPU, Gradio API, and Nomic Atlas to build efficient and innovative applications. From cost-effective GPUs to semantic search, discover how to unlock the full potential of the Hugging Face Hub with this step-by-step guide.

# Model Merging

## [Merge Large Language Models with mergekit, by Maxime Labonne](https://huggingface.co/blog/mlabonne/merge-models)

1. **Merging Large Language Models**: Learn how to use "mergekit" to combine the best features of multiple models, even without access to high-end GPUs. You'll explore advanced techniques like SLERP and TIES, enabling you to create state-of-the-art models that can compete with the best in the field.
   
Unlock the power of large language model merging without GPUs! Discover how to use "mergekit" to combine the best models with ease, explore advanced merging techniques like SLERP and TIES, and learn to create state-of-the-art models that dominate leaderboards—all from your laptop. Dive in to revolutionize your AI capabilities today!

# Training & Finetuning

## [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth, by Maxime Labonne](https://huggingface.co/blog/mlabonne/sft-llama3)

"In this article, we will provide a comprehensive overview of supervised fine-tuning. We will compare it to prompt engineering to understand when it makes sense to use it, detail the main techniques with their pros and cons, and introduce major concepts, such as LoRA hyperparameters, storage formats, and chat templates. Finally, we will implement it in practice by fine-tuning Llama 3.1 8B in Google Colab with state-of-the-art optimization using Unsloth." - Maxime Labonne

## [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft)

Fine-tuning these pretrained LLMs on downstream datasets results in huge performance gains when compared to using the pretrained LLMs out-of-the-box (zero-shot inference, for example).

## [Uncensor any LLM with abliteration, by Maxime Labonne](https://huggingface.co/blog/mlabonne/abliteration)

Discover how to bypass built-in censorship in large language models using "abliteration"! Learn how to unleash the full potential of your LLMs by identifying and removing refusal mechanisms, all without costly retraining. Dive into this step-by-step guide to creating your own uncensored and high-performing AI model—perfect for those looking to unlock unrestricted AI interactions.

## [The case for specialized pre-training: ultra-fast foundation models for dedicated tasks](https://huggingface.co/blog/Pclanglais/specialized-pre-training)

Unlock ultra-fast, dedicated AI with specialized pre-training! Learn how tiny models like OCRonos-Vintage are outshining larger, general-purpose models by focusing on specific tasks. From blazing-fast inference to full control over training data, discover how pre-training for specialized applications can transform performance, save costs, and lead the next wave of efficient AI development.

## [TGI Multi-LoRA: Deploy Once, Serve 30 models](https://huggingface.co/blog/multi-lora-serving)

Are you tired of the complexity and expense of managing multiple AI models? What if you could deploy once and serve 30 models? In today's ML world, organizations looking to leverage the value of their data will likely end up in a fine-tuned world, building a multitude of models, each one highly specialized for a specific task. But how can you keep up with the hassle and cost of deploying a model for each use case? The answer is Multi-LoRA serving.

# Synthetic/Augmented Datasets

This section provides a comprehensive overview of synthetic dataset generation and LLM-driven dataset augmentation techniques. Understanding these concepts is crucial for creating high-quality training data for AI models.

## Key Concepts

1. **Synthetic Dataset Generation**
   The process of using an LLM to generate responses to a series of requests, which are then stored as a dataset.

2. **Synthetic Dataset Collection**
   Utilizing AI assistance in collecting and storing training data.

3. **Data Augmentation**
   An AI assistant uses provided source data (human or AI-generated) as ground truth to create variations such as paraphrases, rephrases, or alternative phrasings.
   
   This process can include:
   
   a) Paraphrasing: Creating synonyms or alternative phrases with the same meaning
   
   b) Expansions: Using paraphrasing to infinitely expand an existing dataset by producing similar data
   
   c) Duplications: Direct data cloning or duplication for alternative applications
   
   d) Insertions: Adding any number of tokens anywhere in the string for direct dataset augmentation
   
   e) Deletions: Removing any number of tokens from anywhere in the string for direct dataset augmentation
   
   f) Formatting: Cleaning and reformatting a dataset to match a provided template during augmentation
   
   g) Configuring: Aligning the training set with user-specified ethics and logical frameworks, potentially using synthetic augmentation for self-improvement under human supervision

5. **Dataset Digestion**
   Involves scraping source data from open-source databases (e.g., arXiv) and tasking an AI agent with constructing a ground truth dataset from the scraped data. This digested data can then be further augmented.

6. **Synthetic Dataset Augmented Expansion**
   The process of expanding existing ground truth data by factors of 10, 100, 1000, etc., through paraphrasing and providing instruction prompt examples. This ensures the AI is modulating the dataset with its knowledge of vocabulary rather than relying on its own knowledge base to ensure we retain the essence of the ground truth data.

## Importance of Synthetic/Augmented Datasets

Synthetic and augmented datasets play a crucial role in AI model development:

1. Increased data quantity: Helps overcome limitations of small datasets

2. Improved diversity: Enhances model generalization and robustness

3. Cost-effective: Reduces the need for expensive manual data collection and labeling

4. Privacy-preserving: Allows for the creation of training data without compromising sensitive information

5. Customization: Enables the generation of data for specific scenarios or edge cases

6. Ethical considerations: Facilitates the creation of datasets that align with desired ethical standards and frameworks

By mastering these techniques, researchers and developers can significantly enhance the quality and effectiveness of their AI models while addressing key challenges in data acquisition and model training.

## [The Rise of Agentic Data Generation, by Maxime Labonne](https://huggingface.co/blog/mlabonne/agentic-datagen)

Explore how innovative frameworks like AgentInstruct and Arena Learning are transforming the creation of high-quality instruction datasets for LLMs. This article delves into these cutting-edge methods, their unique approaches, and the potential of combining them to advance AI training.

## [🦙⚗️ Using Llama3 and distilabel to build fine-tuning datasets](https://huggingface.co/blog/dvilasuero/synthetic-data-with-llama3-distilabel)

The end goal is creating a dataset that can be used to improve OSS models, using alignment methods like DPO, ORPO, or KTO. Now that Llama3 is closing the perfomance gap, we're a step closer to our vision: fully open data generation pipelines!

## [⚗️ 🔥 Building High-Quality Datasets with distilabel and Prometheus 2](https://huggingface.co/blog/burtenshaw/distilabel-prometheus-2)

How to build high-quality datasets for fine-tuning large language models (LLMs) using distilabel and Prometheus 2. Prometheus 2 is an open-source model designed for evaluating LLM generations, providing a cost-effective alternative to GPT-4. This powerful combination allows us to distil both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) datasets efficiently and transparently.

## [How we leveraged distilabel to create an Argilla 2.0 Chatbot](https://huggingface.co/blog/argilla-chatbot)

Discover how to build a Chatbot for a tool of your choice (Argilla 2.0 in this case) that can understand technical documentation and chat with users about it.
In this article, they'll show you how to leverage distilabel and fine-tune a domain-specific embedding model to create a conversational model that's both accurate and engaging.

## [Cosmopedia: how to create large-scale synthetic data for pre-training](https://huggingface.co/blog/cosmopedia)

The challenges and solutions involved in generating a synthetic dataset with billions of tokens to replicate Phi-1.5, leading to the creation of Cosmopedia. Synthetic data has become a central topic in Machine Learning. It refers to artificially generated data, for instance by large language models (LLMs), to mimic real-world data.

## [Synthetic dataset generation techniques: generating custom sentence similarity data](https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets)

One of the most exciting use cases for LLMs is generating synthetic datasets that can be used to train non-LLM models. In the past, gathering enough data was one of the most significant barriers to training task-specific models, such as text classification models. LLMs can potentially help in this area.

## [Docmatix - A huge dataset for document Visual Question Answering](https://huggingface.co/blog/docmatix)

With this blog we are releasing Docmatix - a huge dataset for Document Visual Question Answering (DocVQA) that is 100s of times larger than previously available. Ablations using this dataset for fine-tuning Florence-2 show a 20% increase in performance on DocVQA.

## [zero-shot-vqa-docmatix](https://huggingface.co/blog/zero-shot-vqa-docmatix)

While developing Docmatix, we noticed that fine-tuning Florence-2 on it yielded great performance on DocVQA, but resulted in low scores on the benchmark. To enhance performance, we had to fine-tune the model further on DocVQA to learn the syntax required for the benchmark. Interestingly, this additional fine-tuning seemed to perform worse according to human evaluators, which is why we primarily used it for ablation studies and released the model only trained on Docmatix for broader use.

## [IBM Chain of Thought, Instruction Tuning & More](https://www.ibm.com/topics/instruction-tuning)


## [AI Apps in a Flash with Gradio's Reload Mode](https://huggingface.co/blog/gradio-reload)

How you can build a functional AI application quickly with Gradio's reload mode.

## [Llama can now see and run on your device - welcome Llama 3.2](https://huggingface.co/blog/llama32)

Llama 3.2 is out! Today, we welcome the next iteration of the Llama collection to Hugging Face. This time, we’re excited to collaborate with Meta on the release of multimodal and small models. Ten open-weight models (5 multimodal models and 5 text-only ones) are available on the Hub.

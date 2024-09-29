
# Borch's AI Reading List

<img
src="docs/posters/llama_apple_tree.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

# Getting Started with Hugging Face

## [Total noob’s intro to Hugging Face Transformers](https://huggingface.co/blog/noob_intro_transformers)

Our goal is to demystify what Hugging Face Transformers is and how it works, not to turn you into a machine learning practitioner, but to enable better understanding of and collaboration with those who are.

## [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)

ggml is a machine learning (ML) library written in C and C++ with a focus on Transformer inference. The project is open-source and is being actively developed by a growing community. ggml is similar to ML libraries such as PyTorch and TensorFlow, though it is still in its early stages of development and some of its fundamentals are still changing rapidly.

## [The 5 Most Under-Rated Tools on Hugging Face](https://huggingface.co/blog/unsung-heroes)

Supercharge your AI projects with these hidden gems on Hugging Face! Learn how to use tools like ZeroGPU, Gradio API, and Nomic Atlas to build efficient and innovative applications. From cost-effective GPUs to semantic search, discover how to unlock the full potential of the Hugging Face Hub with this step-by-step guide.

# Model Merging & Quantization

## [Merge Large Language Models with mergekit, by Maxime Labonne](https://huggingface.co/blog/mlabonne/merge-models)

Unlock the power of large language model merging without GPUs! Discover how to use "mergekit" to combine the best models with ease, explore advanced merging techniques like SLERP and TIES, and learn to create state-of-the-art models that dominate leaderboards—all from your laptop. Dive in to revolutionize your AI capabilities today!

## [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)

This article focuses on giving a high-level overview of this quantization technology, outlining the difficulties in incorporating it into the transformers library, and drawing up the long-term goals of this partnership.

# Training & Finetuning

## [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth, by Maxime Labonne](https://huggingface.co/blog/mlabonne/sft-llama3)

"In this article, we will provide a comprehensive overview of supervised fine-tuning. We will compare it to prompt engineering to understand when it makes sense to use it, detail the main techniques with their pros and cons, and introduce major concepts, such as LoRA hyperparameters, storage formats, and chat templates. Finally, we will implement it in practice by fine-tuning Llama 3.1 8B in Google Colab with state-of-the-art optimization using Unsloth." - Maxime Labonne

## [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft)

Fine-tuning these pretrained LLMs on downstream datasets results in huge performance gains when compared to using the pretrained LLMs out-of-the-box (zero-shot inference, for example).

## [Uncensor any LLM with abliteration, by Maxime Labonne](https://huggingface.co/blog/mlabonne/abliteration)

Discover how to bypass built-in censorship in large language models using "abliteration"! Learn how to unleash the full potential of your LLMs by identifying and removing refusal mechanisms, all without costly retraining. Dive into this step-by-step guide to creating your own uncensored and high-performing AI model—perfect for those looking to unlock unrestricted AI interactions.

## [The case for specialized pre-training: ultra-fast foundation models for dedicated tasks](https://huggingface.co/blog/Pclanglais/specialized-pre-training)

Unlock ultra-fast, dedicated AI with specialized pre-training! Learn how tiny models like OCRonos-Vintage are outshining larger, general-purpose models by focusing on specific tasks. From blazing-fast inference to full control over training data, discover how pre-training for specialized applications can transform performance, save costs, and lead the next wave of efficient AI development.

## [Fine-tuning LLMs to 1.58bit: extreme quantization made easy](https://huggingface.co/blog/1_58_llm_extreme_quantization)

BitNet is an architecture introduced by Microsoft Research that uses extreme quantization, representing each parameter with only three values: -1, 0, and 1. This results in a model that uses just 1.58 bits per parameter, significantly reducing computational and memory requirements.

## [TGI Multi-LoRA: Deploy Once, Serve 30 models](https://huggingface.co/blog/multi-lora-serving)

Are you tired of the complexity and expense of managing multiple AI models? What if you could deploy once and serve 30 models? In today's ML world, organizations looking to leverage the value of their data will likely end up in a fine-tuned world, building a multitude of models, each one highly specialized for a specific task. But how can you keep up with the hassle and cost of deploying a model for each use case? The answer is Multi-LoRA serving.

# Synthetic/Augmented datasets

The following resources will provide a good foundational understanding of synthetic dataset generation, as well as llm driven dataset augmentation.
Please go forward with the following understanding:
Synthetic dataset generation typically pertains to asking an llm a series of requests, and storing the responses as a dataset.
Synthetic dataset collection involves ai assistance in collecting/storing training data.
Augmentation occurs when an ai assistant takes the provided source data, human or ai, and uses it as a ground truth to generate paraphrases, rephrases, or alternativeley phrased questions or statements for a training set. Augmentation can involve paraphrases, expansions, duplications, insertions, deletions, formatting, configuring and more as long as its based on ground truth data sources and not just having the ai produce synthetic responses.
Dataset Digestion involves taking/scraping source data from an opensource database such as arxiv, and tasking the agent with constructing the ground truth dataset from the scraped data. This digested data can then be augmented.
Synthetic Dataset Augmented Expansion is the process of expanding existing ground truth data by x10, x100, x1000, etc. via paraphrases and also providing instruction prompt examples to truly seed the dataset and ensure the ai is modulated the provided data, and not pulling from its own knowledge base.

## [The Rise of Agentic Data Generation, by Maxime Labonne](https://huggingface.co/blog/mlabonne/agentic-datagen)

Explore how innovative frameworks like AgentInstruct and Arena Learning are transforming the creation of high-quality instruction datasets for LLMs. This article delves into these cutting-edge methods, their unique approaches, and the potential of combining them to advance AI training.

## [🦙⚗️ Using Llama3 and distilabel to build fine-tuning datasets](https://huggingface.co/blog/dvilasuero/synthetic-data-with-llama3-distilabel)

The end goal is creating a dataset that can be used to improve OSS models, using alignment methods like DPO, ORPO, or KTO. Now that Llama3 is closing the perfomance gap, we're a step closer to our vision: fully open data generation pipelines!

## [⚗️ 🔥 Building High-Quality Datasets with distilabel and Prometheus 2](https://huggingface.co/blog/burtenshaw/distilabel-prometheus-2)

How to build high-quality datasets for fine-tuning large language models (LLMs) using distilabel and Prometheus 2. Prometheus 2 is an open-source model designed for evaluating LLM generations, providing a cost-effective alternative to GPT-4. This powerful combination allows us to distil both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) datasets efficiently and transparently.

## [How we leveraged distilabel to create an Argilla 2.0 Chatbot](https://huggingface.co/blog/argilla-chatbot)

Discover how to build a Chatbot for a tool of your choice (Argilla 2.0 in this case) that can understand technical documentation and chat with users about it.
In this article, we'll show you how to leverage distilabel and fine-tune a domain-specific embedding model to create a conversational model that's both accurate and engaging.

## [Cosmopedia: how to create large-scale synthetic data for pre-training](https://huggingface.co/blog/cosmopedia)

The challenges and solutions involved in generating a synthetic dataset with billions of tokens to replicate Phi-1.5, leading to the creation of Cosmopedia. Synthetic data has become a central topic in Machine Learning. It refers to artificially generated data, for instance by large language models (LLMs), to mimic real-world data.

## [Synthetic dataset generation techniques: generating custom sentence similarity data](https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets)

## [Docmatix - A huge dataset for document Visual Question Answering](https://huggingface.co/blog/docmatix)

## [zero-shot-vqa-docmatix](https://huggingface.co/blog/zero-shot-vqa-docmatix)

# Model Deployment, Agents, Chatbots, and guides

## [Tool Use, Unified](https://huggingface.co/blog/unified-tool-use)

## [AI Apps in a Flash with Gradio's Reload Mode](https://huggingface.co/blog/gradio-reload)

## [Llama can now see and run on your device - welcome Llama 3.2](https://huggingface.co/blog/llama32)


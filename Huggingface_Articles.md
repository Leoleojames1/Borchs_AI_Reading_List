# Getting Started with Hugging Face

## [Total noob’s intro to Hugging Face Transformers](https://huggingface.co/blog/noob_intro_transformers)

## [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)

## [The 5 Most Under-Rated Tools on Hugging Face](https://huggingface.co/blog/unsung-heroes)

Supercharge your AI projects with these hidden gems on Hugging Face! Learn how to use tools like ZeroGPU, Gradio API, and Nomic Atlas to build efficient and innovative applications. From cost-effective GPUs to semantic search, discover how to unlock the full potential of the Hugging Face Hub with this step-by-step guide.

# Model Merging & Quantization

## [Merge Large Language Models with mergekit, by Maxime Labonne](https://huggingface.co/blog/mlabonne/merge-models)

Unlock the power of large language model merging without GPUs! Discover how to use "mergekit" to combine the best models with ease, explore advanced merging techniques like SLERP and TIES, and learn to create state-of-the-art models that dominate leaderboards—all from your laptop. Dive in to revolutionize your AI capabilities today!

# Training & Finetuning

## [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft)

## [Uncensor any LLM with abliteration, by Maxime Labonne](https://huggingface.co/blog/mlabonne/abliteration)

Discover how to bypass built-in censorship in large language models using "abliteration"! Learn how to unleash the full potential of your LLMs by identifying and removing refusal mechanisms, all without costly retraining. Dive into this step-by-step guide to creating your own uncensored and high-performing AI model—perfect for those looking to unlock unrestricted AI interactions.

## [The case for specialized pre-training: ultra-fast foundation models for dedicated tasks](https://huggingface.co/blog/Pclanglais/specialized-pre-training)

Unlock ultra-fast, dedicated AI with specialized pre-training! Learn how tiny models like OCRonos-Vintage are outshining larger, general-purpose models by focusing on specific tasks. From blazing-fast inference to full control over training data, discover how pre-training for specialized applications can transform performance, save costs, and lead the next wave of efficient AI development.

## [Fine-tuning LLMs to 1.58bit: extreme quantization made easy](https://huggingface.co/blog/1_58_llm_extreme_quantization)

## [TGI Multi-LoRA: Deploy Once, Serve 30 models](https://huggingface.co/blog/multi-lora-serving)

# Model Deployment, Agents, Chatbots, and guides

## [Tool Use, Unified](https://huggingface.co/blog/unified-tool-use)

## [AI Apps in a Flash with Gradio's Reload Mode](https://huggingface.co/blog/gradio-reload)

## [Llama can now see and run on your device - welcome Llama 3.2](https://huggingface.co/blog/llama32)

# Synthetic/Augmented datasets

The following resources will provide a good foundational understanding of synthetic dataset generation, as well as llm driven dataset augmentation.
Please go forward with the following understanding:
Synthetic dataset generation typically pertains to asking an llm a series of requests, and storing the responses as a dataset.
Synthetic dataset collection involves ai assistance in collecting/storing training data.
Augmentation occurs when an ai assistant takes the provided source data, human or ai, and uses it as a ground truth to generate paraphrases, rephrases, or alternativeley phrased questions or statements for a training set. Augmentation can involve paraphrases, expansions, duplications, insertions, deletions, formatting, configuring and more as long as its based on ground truth data sources and not just having the ai produce synthetic responses.
Dataset Digestion involves taking/scraping source data from an opensource database such as arxiv, and tasking the agent with constructing the ground truth dataset from the scraped data. This digested data can then be augmented.
Synthetic Dataset Augmented Expansion is the process of expanding existing ground truth data by x10, x100, x1000, etc. via paraphrases and also providing instruction prompt examples to truly seed the dataset and ensure the ai is modulated the provided data, and not pulling from its own knowledge base.

## [The Rise of Agentic Data Genation, by Maxime Labonne](https://huggingface.co/blog/mlabonne/agentic-datagen)

Explore how innovative frameworks like AgentInstruct and Arena Learning are transforming the creation of high-quality instruction datasets for LLMs. This article delves into these cutting-edge methods, their unique approaches, and the potential of combining them to advance AI training.

## [🦙⚗️ Using Llama3 and distilabel to build fine-tuning datasets](https://huggingface.co/blog/dvilasuero/synthetic-data-with-llama3-distilabel)

## [Cosmopedia: how to create large-scale synthetic data for pre-training](https://huggingface.co/blog/cosmopedia)

## [Synthetic dataset generation techniques: generating custom sentence similarity data](https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets)

## [Docmatix - A huge dataset for document Visual Question Answering](https://huggingface.co/blog/docmatix)

## [zero-shot-vqa-docmatix](https://huggingface.co/blog/zero-shot-vqa-docmatix)


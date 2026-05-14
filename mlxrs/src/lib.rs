#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

/// Audio Language Model (ALM) is a type of artificial intelligence model that is designed to understand and generate human language in the context of audio data.
/// It is trained on large datasets of audio recordings and their corresponding transcriptions, allowing it to perform various tasks such as speech recognition, audio transcription, and audio-based natural language processing.
/// ALMs are commonly used in applications like voice assistants, transcription services, and audio-based chatbots.
#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio;

/// Language Model (LM) is a type of artificial intelligence model that is designed to understand and generate human language.
/// It is trained on large datasets of text and can perform various natural language processing tasks,
/// such as text generation, translation, summarization, and question answering.
/// LMs are commonly used in applications like chatbots, virtual assistants,
/// and language translation services.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
pub mod lm;

/// Vector Language Model (VLM) is a language model that can process both text and images.
/// It is designed to understand and generate language in the context of visual information,
/// making it useful for tasks that involve both modalities, such as image captioning,
/// visual question answering, and multimodal content generation.
#[cfg(feature = "vlm")]
#[cfg_attr(docsrs, doc(cfg(feature = "vlm")))]
pub mod vlm;

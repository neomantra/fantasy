# Providers

Fantasy ships with a catalog of model providers. Pick one, plug in credentials,
and point it at your favorite model.

| Provider | Package | Description | Setup |
| --- | --- | --- | --- |
| Anthropic | [`providers/anthropic`](./anthropic) | Claude models via the official Anthropic API. | — |
| Azure | [`providers/azure`](./azure) | Azure-hosted OpenAI models. | [README](./azure/README.md) |
| AWS Bedrock | [`providers/bedrock`](./bedrock) | Anthropic models served through AWS Bedrock. | [README](./bedrock/README.md) |
| Google | [`providers/google`](./google) | Gemini via AI Studio and Vertex AI. | [README](./google/README.md) |
| Kronk | [`providers/kronk`](./kronk) | Local, hardware-accelerated inference with [Kronk](https://github.com/ardanlabs/kronk) (experimental). | [README](./kronk/README.md) |
| OpenAI | [`providers/openai`](./openai) | OpenAI's Chat Completions and Responses APIs. | — |
| OpenAI-compatible | [`providers/openaicompat`](./openaicompat) | Generic layer for any OpenAI-compatible endpoint (Groq, Together, Fireworks, Ollama, vLLM, and friends). | — |
| OpenRouter | [`providers/openrouter`](./openrouter) | Unified access to hundreds of models through [OpenRouter](https://openrouter.ai). | — |
| Vercel AI Gateway | [`providers/vercel`](./vercel) | Routing, observability, and key management via the Vercel AI Gateway. | — |

## Don't see your provider?

If it speaks the OpenAI API, `openaicompat` probably has you covered — point it
at the base URL and go. If your provider needs special treatment, open an issue
or a PR.

## Examples

Runnable examples for several providers live in
[`examples/`](../examples).

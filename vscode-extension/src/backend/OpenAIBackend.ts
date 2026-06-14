import { BaseLLM, CompletionRequest, CompletionResponse, StreamChunk } from "./BaseLLM";

export class OpenAIBackend extends BaseLLM {
    private model: string = "gpt-4";

    constructor(apiKey: string, model?: string) {
        super();
        this.apiKey = apiKey;
        if (model) { this.model = model; }
    }

    async complete(request: CompletionRequest): Promise<CompletionResponse> {
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [{ role: "user", content: request.prompt }],
                max_tokens: request.maxTokens || 100,
                temperature: request.temperature || 0.7,
                stream: false
            })
        });

        const data = await response.json();
        const message = data.choices[0]?.message?.content || "";

        return {
            text: message,
            tokens: [], // OpenAI不返回token列表
            usage: data.usage
        };
    }

    async *stream(request: CompletionRequest): AsyncGenerator<StreamChunk> {
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [{ role: "user", content: request.prompt }],
                max_tokens: request.maxTokens || 100,
                temperature: request.temperature || 0.7,
                stream: true
            })
        });

        const reader = response.body?.getReader();
        if (!reader) { return; }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) { break; }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = line.slice(6);
                    if (data === "[DONE]") {
                        yield { text: "", done: true };
                        return;
                    }
                    const parsed = JSON.parse(data);
                    const text = parsed.choices[0]?.delta?.content || "";
                    if (text) {
                        yield { text, done: false };
                        this.emit("token", text);
                    }
                }
            }
        }

        yield { text: "", done: true };
    }

    supportsStreaming(): boolean {
        return true;
    }
}

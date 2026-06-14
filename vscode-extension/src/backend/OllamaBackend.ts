import { BaseLLM, CompletionRequest, CompletionResponse, StreamChunk } from "./BaseLLM";

export class OllamaBackend extends BaseLLM {
    private _baseURL: string;
    private _model: string;

    constructor(baseURL: string = "http://localhost:11434", model: string = "codellama") {
        super();
        this._baseURL = baseURL;
        this._model = model;
    }

    async complete(request: CompletionRequest): Promise<CompletionResponse> {
        const response = await fetch(`${this._baseURL}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: this._model,
                prompt: request.prompt,
                options: {
                    temperature: request.temperature || 0.7,
                    num_predict: request.maxTokens || 100
                },
                stream: false
            })
        });

        const data = await response.json();
        return {
            text: data.response || "",
            tokens: []
        };
    }

    async *stream(request: CompletionRequest): AsyncGenerator<StreamChunk> {
        const response = await fetch(`${this._baseURL}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: this._model,
                prompt: request.prompt,
                options: {
                    temperature: request.temperature || 0.7,
                    num_predict: request.maxTokens || 100
                },
                stream: true
            })
        });

        const reader = response.body?.getReader();
        if (!reader) { return; }

        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) { break; }

            const text = decoder.decode(value, { stream: true });
            const lines = text.split("\n").filter(l => l.trim());

            for (const line of lines) {
                try {
                    const data = JSON.parse(line);
                    if (data.response) {
                        yield { text: data.response, done: false };
                        this.emit("token", data.response);
                    }
                    if (data.done) {
                        yield { text: "", done: true };
                        return;
                    }
                } catch {}
            }
        }

        yield { text: "", done: true };
    }

    supportsStreaming(): boolean {
        return true;
    }
}

import { EventEmitter } from "events";

export interface CompletionRequest {
    prompt: string;
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stop?: string[];
}

export interface CompletionResponse {
    text: string;
    tokens: number[];
    logits?: number[][];
    usage?: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
    };
}

export interface StreamChunk {
    text: string;
    done: boolean;
}

export abstract class BaseLLM extends EventEmitter {
    protected apiKey?: string;
    protected baseURL?: string;

    abstract complete(request: CompletionRequest): Promise<CompletionResponse>;

    abstract stream(request: CompletionRequest): AsyncGenerator<StreamChunk>;

    abstract supportsStreaming(): boolean;

    protected getConfig(): Record<string, any> {
        return {};
    }
}

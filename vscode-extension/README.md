# MetaCog-X VSCode Extension

Conditional Metacognition for AI Code Generation.

## Features

- **Dilemma Detection**: Detects code generation困境 (semantic stuck, syntax anomalies, pattern repetition)
- **Smart Intervention**: Automatically triggers correction strategies (backtrack, rerank, diversity boost)
- **Multi-Backend Support**: Works with OpenAI, Claude, Ollama, and vLLM
- **Status Bar**: Visual indicator of current generation mode
- **Configurable Thresholds**: Fine-tune detection sensitivity

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Compile TypeScript:
   ```bash
   npm run compile
   ```
4. Package the extension:
   ```bash
   npx vsce package
   ```
5. Install the .vsix file in VS Code:
   ```bash
   code --install-extension metacog-x-0.1.0.vsix
   ```

## Configuration

Configure in VS Code Settings (JSON):

```json
{
    "metacog-x.enabled": true,
    "metacog-x.llmProvider": "ollama",
    "metacog-x.baseURL": "http://localhost:11434",
    "metacog-x.model": "codellama",
    "metacog-x.thresholds": {
        "tokenRepetition": 0.3,
        "logitsEntropy": 2.5,
        "ngramRepetition": 0.4,
        "consecutiveSame": 3
    }
}
```

## Commands

| Command | Description |
|---------|-------------|
| `metacog-x.showStatus` | Show current status |
| `metacog-x.toggle` | Toggle extension on/off |
| `metacog-x.showSettings` | Open settings |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  VS Code IDE                     │
│  ┌─────────────────────────────────────────┐    │
│  │         MetaCog-X Extension              │    │
│  │  ┌──────────┐  ┌─────────┐  ┌────────┐ │    │
│  │  │ Dilemma  │  │Strategy │  │LLM     │ │    │
│  │  │Detector  │──│Router   │──│Backend │ │    │
│  │  └──────────┘  └─────────┘  └────────┘ │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Requirements

- VS Code 1.80.0+
- Node.js 18+
- LLM backend (Ollama, OpenAI API, etc.)

## License

MIT

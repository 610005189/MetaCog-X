# Contributing to MetaCog-X

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Compile TypeScript:
   ```bash
   npm run compile
   ```

## Testing

Run tests:
```bash
npm test
```

## Building

Package the extension:
```bash
npx vsce package
```

## Publishing

1. Create a publisher account at https://marketplace.visualstudio.com/
2. Login:
   ```bash
   npx vsce login <publisher>
   ```
3. Publish:
   ```bash
   npx vsce publish
   ```

## Code Structure

- `src/detector/` - Dilemma detection module
- `src/strategy/` - Intervention strategies
- `src/backend/` - LLM backend integrations
- `src/ui/` - UI components

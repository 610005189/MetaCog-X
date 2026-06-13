# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

If you discover a security vulnerability in MetaCog-X, please follow these steps:

1. **Do not open a public issue** - Security vulnerabilities should be reported privately to avoid exposing the issue before a fix is available.

2. **Contact us directly** - Send an email to [security@metacog-x.dev](mailto:security@metacog-x.dev) with the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (optional)

3. **Expect a response** - We aim to acknowledge security reports within 48 hours and will work with you to understand and fix the issue.

4. **Disclosure** - Once the vulnerability has been fixed, we will coordinate a public disclosure timeline with you.

## Security Best Practices

When using MetaCog-X:

- Keep dependencies up to date
- Use secure environments for training and inference
- Be cautious when loading external model weights
- Validate all inputs before processing

## Known Security Considerations

- Model weights may contain biases or harmful content
- Large language models can generate harmful outputs
- Always implement appropriate input validation and filtering

---

*Last updated: June 2026*
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | ✅ Active support  |
| 1.2.x   | ✅ Security fixes  |
| < 1.2   | ❌ No support      |

## Reporting a Vulnerability

We take security seriously at AutoML Studio Pro. If you discover a security vulnerability, please follow responsible disclosure practices.

### How to Report

**Email:** himanshu231204@gmail.com

**Subject:** `[SECURITY] AutoML Studio Pro - Vulnerability Report`

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if any)
- Your contact information for follow-up

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| **Acknowledgment** | Within 48 hours |
| **Initial Assessment** | Within 1 week |
| **Fix Development** | 2-4 weeks (depending on severity) |
| **Public Disclosure** | After fix is released |

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| 🔴 **Critical** | Remote code execution, data breach | 24-48 hours |
| 🟠 **High** | Privilege escalation, significant data exposure | 1 week |
| 🟡 **Medium** | Limited data exposure, requires specific conditions | 2 weeks |
| 🟢 **Low** | Minor issues, best practice violations | 1 month |

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version
2. **Secure Environment**: Run in isolated environments
3. **Data Privacy**: Don't upload sensitive data to demo instances
4. **Model Security**: Protect exported models from unauthorized access

### For Developers

1. **Input Validation**: Always validate user inputs
2. **Dependency Scanning**: Use `safety` to check for vulnerable packages
3. **Code Review**: All changes require review before merge
4. **Secrets Management**: Never commit credentials or API keys

## Security Features

### Built-in Protections

- ✅ XSRF protection enabled
- ✅ Input sanitization for CSV uploads
- ✅ Model validation before loading
- ✅ Secure file handling
- ✅ Dependency vulnerability scanning in CI

### Data Handling

- Uploaded files are processed in memory
- No data is stored permanently on servers
- Model artifacts are user-controlled
- Session data is cleared on browser close

## Dependency Security

We use automated tools to monitor dependencies:

- **Safety**: Scans for known vulnerabilities
- **Dependabot**: Automated dependency updates
- **GitHub Security Advisories**: Monitored for new vulnerabilities

## Contact

For security-related inquiries:

- **Email:** himanshu231204@gmail.com
- **GitHub Security:** [Create a security advisory](https://github.com/himanshu231204/AutoML-Studio-Pro-/security/advisories/new)

---

**Last Updated:** March 2026

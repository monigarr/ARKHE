# Security Best Practices

This guide covers security best practices for the ARKHE Framework, including dependency management, secrets handling, and code security.

## Table of Contents

1. [Overview](#overview)
2. [Dependency Security](#dependency-security)
3. [Secrets Management](#secrets-management)
4. [Code Security](#code-security)
5. [Security Scanning](#security-scanning)
6. [Reporting Vulnerabilities](#reporting-vulnerabilities)

## Overview

Security is a critical aspect of enterprise software. This guide provides best practices for:
- Managing dependencies securely
- Handling secrets and sensitive data
- Writing secure code
- Automated security scanning

## Dependency Security

### Vulnerability Scanning

The project includes automated dependency vulnerability scanning:

#### Using Safety

```bash
# Install safety
pip install safety

# Scan requirements.txt
safety check --file requirements.txt

# Scan requirements-dev.txt
safety check --file requirements-dev.txt
```

#### Using pip-audit

```bash
# Install pip-audit
pip install pip-audit

# Audit requirements
pip-audit --requirement requirements.txt

# Generate JSON report
pip-audit --requirement requirements.txt --format json --output report.json
```

#### Automated Scanning

Run the security scan script:

```bash
# Run all security scans
python scripts/security_scan.py --all

# Run only dependency scans
python scripts/security_scan.py --dependencies
```

### Keeping Dependencies Updated

1. **Regular Updates**: Review and update dependencies monthly
2. **Security Advisories**: Subscribe to security advisories for key dependencies
3. **Pin Versions**: Use specific versions in production (not `>=`)
4. **Test Updates**: Always test after updating dependencies

### Dependency Management Best Practices

```bash
# Use requirements.txt with minimum versions
numpy>=1.24.0

# For production, consider pinning exact versions
numpy==1.24.0

# Regularly update
pip install --upgrade package-name
pip freeze > requirements.txt
```

## Secrets Management

### Never Commit Secrets

**⚠️ CRITICAL: Never commit secrets, API keys, passwords, or tokens to version control!**

### Using Environment Variables

```python
import os

# Good: Read from environment variable
api_key = os.getenv("ARKHE_API_KEY")
if not api_key:
    raise ValueError("ARKHE_API_KEY environment variable not set")
```

### Using .env Files (Development Only)

Create a `.env` file (and add it to `.gitignore`):

```bash
# .env (DO NOT COMMIT THIS FILE!)
ARKHE_API_KEY=your-api-key-here
DATABASE_PASSWORD=your-password-here
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

api_key = os.getenv("ARKHE_API_KEY")
```

### Production Secrets Management

For production, use proper secret management:

- **AWS Secrets Manager**: For AWS deployments
- **HashiCorp Vault**: Enterprise secret management
- **Kubernetes Secrets**: For containerized deployments
- **Environment Variables**: Set by deployment system

Example with AWS Secrets Manager:

```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret("arkhe/production/api-keys")
api_key = secrets["ARKHE_API_KEY"]
```

### Secrets Detection

The project uses Gitleaks to detect secrets in code:

```bash
# Run Gitleaks scan
gitleaks detect --source . --verbose

# Or use GitHub Actions (automated)
# See .github/workflows/security-scan.yml
```

## Code Security

### Input Validation

Always validate user inputs:

```python
from math_research.utils.validators import validate_positive_int

# Validate input
try:
    start_value = validate_positive_int(user_input, "start_value")
except ValueError as e:
    # Handle validation error
    logger.error(f"Invalid input: {e}")
    return
```

### SQL Injection Prevention

If using databases, use parameterized queries:

```python
# Good: Parameterized query
cursor.execute("SELECT * FROM sequences WHERE start = %s", (start_value,))

# Bad: String concatenation (vulnerable to SQL injection)
cursor.execute(f"SELECT * FROM sequences WHERE start = {start_value}")
```

### Path Traversal Prevention

When handling file paths:

```python
from pathlib import Path

# Good: Use Path and resolve
user_path = Path(user_input).resolve()
base_dir = Path("/app/data").resolve()

# Ensure path is within base directory
if not str(user_path).startswith(str(base_dir)):
    raise ValueError("Path outside allowed directory")
```

### Code Security Scanning

#### Using Bandit

```bash
# Install bandit
pip install bandit[toml]

# Scan source code
bandit -r src/

# Generate JSON report
bandit -r src/ -f json -o bandit-report.json

# Scan with specific severity level
bandit -r src/ -ll  # Low and above
bandit -r src/ -lll  # Medium and above
```

#### Automated Scanning

```bash
# Run code security scan
python scripts/security_scan.py --code
```

## Security Scanning

### Automated CI/CD Scanning

The project includes GitHub Actions workflows for automated security scanning:

- **Dependency Scanning**: Runs on every push and weekly
- **Code Scanning**: Runs on every push
- **Secrets Scanning**: Runs on every push

See `.github/workflows/security-scan.yml` for details.

### Manual Scanning

Run security scans locally:

```bash
# Run all scans
python scripts/security_scan.py --all

# Or run individually
python scripts/security_scan.py --dependencies
python scripts/security_scan.py --code
```

### Pre-commit Hooks

Add security checks to pre-commit hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-r", "src/"]
```

## Reporting Vulnerabilities

If you discover a security vulnerability:

1. **DO NOT** create a public GitHub issue
2. **DO** email: monigarr@MoniGarr.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

See [SECURITY.md](../../SECURITY.md) for the full vulnerability reporting process.

## Security Checklist

Before deploying to production:

- [ ] All dependencies scanned for vulnerabilities
- [ ] No secrets committed to version control
- [ ] Environment variables used for sensitive data
- [ ] Input validation implemented
- [ ] Code security scan passed
- [ ] Security headers configured (if web app)
- [ ] HTTPS enabled (if web app)
- [ ] Rate limiting implemented (if API)
- [ ] Logging configured (no sensitive data in logs)
- [ ] Backup encryption enabled

## Additional Resources

- [SECURITY.md](../../SECURITY.md) - Security policy and vulnerability reporting
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Common security risks
- [Python Security](https://python.readthedocs.io/en/latest/library/security.html) - Python security best practices
- [Bandit Documentation](https://bandit.readthedocs.io/) - Code security scanner
- [Safety Documentation](https://pyup.io/safety/) - Dependency vulnerability scanner


# Security Policy

## 🔒 Repository Protection

This repository has multiple layers of protection to prevent unauthorized access and unwanted contributions.

### Branch Protection Rules

- **Main Branch**: Protected with required pull request reviews
- **Force Push**: Disabled
- **Branch Deletion**: Disabled
- **Admin Override**: Enabled (for repository owner)

### User Blocking Mechanisms

#### 1. GitHub Actions Workflow
- **File**: `.github/workflows/block-user.yml`
- **Triggers**: On push and pull requests
- **Blocks**: 
  - Email patterns containing "questionpro.com" or "qp"
  - Name patterns containing "ritwik-yadav-qp" or "questionpro"

#### 2. Local Git Hooks
- **File**: `.git/hooks/pre-push`
- **Purpose**: Prevents local pushes from unwanted users
- **Blocks**: Same patterns as GitHub Actions

### How It Works

1. **Pre-push Hook**: Checks commit author before pushing locally
2. **GitHub Actions**: Validates commits on the server side
3. **Branch Protection**: Requires pull request reviews for main branch
4. **Admin Override**: Repository owner can bypass protections when needed

### Bypassing Protections

Only the repository owner (`delirium0712`) can:
- Bypass branch protection rules
- Push directly to main branch
- Override user blocking mechanisms

### Monitoring

All blocked attempts are logged and visible in:
- GitHub Actions logs
- Local git hook output
- Repository security settings

## 🛡️ Security Best Practices

1. **Never commit sensitive data** (API keys, passwords, etc.)
2. **Use pull requests** for all changes
3. **Review all contributions** before merging
4. **Monitor repository access** regularly
5. **Keep dependencies updated** for security patches

## 🚨 Incident Response

If you notice unauthorized access or suspicious activity:
1. Check GitHub Actions logs
2. Review recent commits and pull requests
3. Update protection rules if needed
4. Contact GitHub support if necessary

---

*This security policy is automatically enforced by the repository's protection mechanisms.*

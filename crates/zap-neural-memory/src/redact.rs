//! 敏感凭据脱敏模块：防止 API Key、Bearer Token、密码、私钥等明文进入记忆库或备份导出。

use std::sync::OnceLock;

use regex::Regex;

use super::types::RawTurn;

struct RedactRule {
    re: Regex,
    replacement: &'static str,
}

fn rules() -> &'static [RedactRule] {
    static RULES: OnceLock<Vec<RedactRule>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            // PEM 私钥块
            RedactRule {
                re: Regex::new(
                    r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                )
                .unwrap(),
                replacement: "[REDACTED_PRIVATE_KEY]",
            },
            // Bearer / Authorization Token
            RedactRule {
                re: Regex::new(r"(?i)\b(Bearer\s+)[A-Za-z0-9\-._~+/]+=*").unwrap(),
                replacement: "${1}[REDACTED]",
            },
            // 常见云厂商 / AI / Git 令牌前缀
            RedactRule {
                re: Regex::new(
                    r"\b(?:sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{16,}|AKIA[0-9A-Z]{16})\b",
                )
                .unwrap(),
                replacement: "[REDACTED_KEY]",
            },
            // URL 内嵌 user:password@host 凭据
            RedactRule {
                re: Regex::new(r"(?i)(https?://[^:\s/@]+:)[^@\s/]+(@)").unwrap(),
                replacement: "${1}[REDACTED]${2}",
            },
            // 键值对密码 / 密钥 / Token
            RedactRule {
                re: Regex::new(
                    r#"(?i)\b(password|passwd|secret|api_?key|access_?token|refresh_?token|auth_?token|private_?key)\s*([=:])\s*['"]?([^\s'"`,;]{6,})['"]?"#,
                )
                .unwrap(),
                replacement: "${1}${2}[REDACTED]",
            },
        ]
    })
}

/// 对文本中的常见密钥与口令进行脱敏替换。
pub(crate) fn redact_secrets(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut out = input.to_string();
    for rule in rules() {
        out = rule.re.replace_all(&out, rule.replacement).into_owned();
    }
    out
}

pub(crate) fn redact_raw_turn(mut turn: RawTurn) -> RawTurn {
    turn.user_text = redact_secrets(&turn.user_text);
    turn.ai_text = redact_secrets(&turn.ai_text);
    turn
}

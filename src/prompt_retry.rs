//! Context-overflow retry from upstream keryx-miner v0.6.2-PoM, src/llm.rs
//! (3e82e42c1b8e8195c50ae0b158ac1bd17b33f6d1). No proactive shortening: the caller
//! must first receive PromptTooLong from a traced engine on its leased card.

/// Context tokens the node reserves beyond the prompt and the answer budget.
const CONTEXT_OVERHEAD_TOKENS: usize = 512;

fn prompt_budget(ctx_floor: u32, max_tokens: usize) -> usize {
    (ctx_floor as usize).saturating_sub(max_tokens.saturating_add(CONTEXT_OVERHEAD_TOKENS))
}

/// Upstream's exact UTF-8 head/tail cut, including its five-byte separator.
fn head_tail_cut(prompt: &str, cap: usize) -> Option<String> {
    const SEP: &str = "\n…\n";
    if prompt.len() <= cap {
        return None;
    }
    let budget = cap.saturating_sub(SEP.len());
    let mut head = budget / 2;
    while !prompt.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = prompt.len() - (budget - head);
    while !prompt.is_char_boundary(tail) {
        tail += 1;
    }
    Some(if cap < SEP.len() { String::new() } else { format!("{}{SEP}{}", &prompt[..head], &prompt[tail..]) })
}

pub(crate) fn retry_prompt(
    prompt: &str,
    ctx_floor: u32,
    max_tokens: usize,
    original_prompt_len: u32,
) -> Option<String> {
    let cap = prompt_budget(ctx_floor, max_tokens);
    if original_prompt_len as usize > cap {
        return None;
    }
    head_tail_cut(prompt, cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_large_prompts_retry_but_large_original_requests_do_not() {
        for size in [74_739, 79_150, 70 * 1024, 100 * 1024, 250 * 1024, 500 * 1024, 1 << 20] {
            let text = format!("HEAD{}TAIL", "x".repeat(size - 8));
            let cut = retry_prompt(&text, 32_768, 128, 25_000).unwrap();
            assert_eq!(cut.len(), 32_128);
            assert!(cut.starts_with("HEAD") && cut.ends_with("TAIL") && cut.contains("\n…\n"));
            assert!(retry_prompt(&text, 32_768, 128, 32_129).is_none());
        }
    }

    #[test]
    fn retries_respect_unicode_boundaries_and_upstream_rounding() {
        let text = "αβ🙂漢字\0".repeat(100);
        for cap in 0..text.len() + 1 {
            let cut = head_tail_cut(&text, cap);
            if cap >= text.len() {
                assert!(cut.is_none());
                continue;
            }
            let cut = cut.unwrap();
            assert!(cut.len() <= cap);
            if cap < 5 {
                assert!(cut.is_empty());
                continue;
            }
            let (head, tail) = cut.split_once("\n…\n").unwrap();
            assert!(text.starts_with(head) && text.ends_with(tail));
            // The explicit-length FFI already supports NUL; never regress it by stripping bytes.
            if head.contains('\0') || tail.contains('\0') {
                assert!(cut.contains('\0'));
            }
        }
    }

    #[test]
    fn budget_saturates_and_original_length_gate_is_inclusive() {
        assert_eq!(prompt_budget(32_768, 128), 32_128);
        assert_eq!(prompt_budget(512, usize::MAX), 0);
        assert_eq!(prompt_budget(0, 0), 0);
        let text = "x".repeat(32_129);
        assert!(retry_prompt(&text, 32_768, 128, 32_128).is_some());
        assert!(retry_prompt(&text, 32_768, 128, 32_129).is_none());
        assert!(retry_prompt("small", 32_768, 128, 5).is_none());
    }
}

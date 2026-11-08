//! Prompt text handed to the native engines' traced-generation C ABI.
//!
//! The node accepts an AiRequest whose prompt contains NUL bytes, and the trace must commit the
//! tokens of the EXACT prompt (the verifier bounds `n_prompt` by the prompt's byte length and the
//! audit replays the committed token ids). A `CString` / `strlen` hand-off cannot carry such a
//! prompt, so engines that export `keryx_llama_generate_traced_n` (0.16.2+) get the bytes with an
//! explicit length. Older libraries only have the NUL-terminated entry point: a prompt with a NUL
//! byte is refused there (never stripped or replaced — that would answer a different prompt).
use std::ffi::{c_char, CString};

/// The `(prompt, rest)` arguments of one traced generation, in the form the loaded library takes.
pub enum TracedPromptArgs<'a> {
    /// Length-aware entry point: the exact bytes, NULs included.
    Sized { prompt: &'a [u8], rest: &'a [u8] },
    /// Pre-0.16.2 library: NUL-terminated copies (only for prompts without a NUL byte).
    Terminated { prompt: CString, rest: CString },
}

impl<'a> TracedPromptArgs<'a> {
    /// `sized`: the library exports the length-aware entry point.
    pub fn new(prompt: &'a str, rest: &'a str, sized: bool) -> Result<Self, String> {
        if sized {
            return Ok(Self::Sized { prompt: prompt.as_bytes(), rest: rest.as_bytes() });
        }
        let nul = || {
            "prompt contains a NUL byte and this engine library predates length-aware traced generation \
             (keryx_llama_generate_traced_n) — update the engine library"
                .to_string()
        };
        Ok(Self::Terminated { prompt: CString::new(prompt).map_err(|_| nul())?, rest: CString::new(rest).map_err(|_| nul())? })
    }

    /// Pointer and byte length of the prompt (for `Terminated`, the length strlen() will see).
    pub fn prompt(&self) -> (*const c_char, usize) {
        match self {
            Self::Sized { prompt, .. } => (prompt.as_ptr() as *const c_char, prompt.len()),
            Self::Terminated { prompt, .. } => (prompt.as_ptr(), prompt.as_bytes().len()),
        }
    }

    /// Pointer and byte length of the rest after the request prefix.
    pub fn rest(&self) -> (*const c_char, usize) {
        match self {
            Self::Sized { rest, .. } => (rest.as_ptr() as *const c_char, rest.len()),
            Self::Terminated { rest, .. } => (rest.as_ptr(), rest.as_bytes().len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nul_prompt_is_passed_with_its_full_length() {
        let head = "<|im_start|>system\nRequest id:";
        let rest = "\nsys<|im_end|>\n<|im_start|>user\nab\0cd\0<|im_end|>\n";
        let args = TracedPromptArgs::new(head, rest, true).unwrap();
        let (pp, pl) = args.prompt();
        let (rp, rl) = args.rest();
        assert_eq!((pp as *const u8, pl), (head.as_ptr(), head.len()));
        assert_eq!((rp as *const u8, rl), (rest.as_ptr(), rest.len()), "the full rest, past both NULs");
        let seen = unsafe { std::slice::from_raw_parts(rp as *const u8, rl) };
        assert_eq!(seen, rest.as_bytes(), "bytes are passed verbatim, never stripped or replaced");
        assert_eq!(seen.iter().filter(|&&b| b == 0).count(), 2);
    }

    #[test]
    fn nul_prompt_is_refused_by_an_old_library() {
        assert!(TracedPromptArgs::new("head", "a\0b", false).is_err());
        assert!(TracedPromptArgs::new("h\0ead", "ab", false).is_err());
    }

    #[test]
    fn old_library_gets_terminated_copies_of_a_clean_prompt() {
        let args = TracedPromptArgs::new("head", "rest", false).unwrap();
        let (pp, pl) = args.prompt();
        let (rp, rl) = args.rest();
        assert_eq!((pl, rl), (4, 4));
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(pp).to_bytes(), b"head");
            assert_eq!(std::ffi::CStr::from_ptr(rp).to_bytes(), b"rest");
        }
    }

    #[test]
    fn empty_rest_is_zero_length() {
        let args = TracedPromptArgs::new("whole prompt", "", true).unwrap();
        assert_eq!(args.rest().1, 0);
        assert_eq!(args.prompt().1, 12);
    }
}

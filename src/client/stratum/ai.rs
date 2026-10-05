//! `mining.ai_request` / `mining.ai_response` — pool-dispatched AI inference (upstream
//! keryx-miner 2cf7ab5, neuropool protocol). From the H14 gate every on-chain request is a
//! private envelope sealed to the tier cohort's escrow keys; in pool mining that cohort member is
//! the POOL (its escrow key signs the coinbase), so a v3 pool opens the envelope itself, sends the
//! plaintext prompt here, and seals + signs the answer it gets back. The miner never sees a key
//! and nothing is published to IPFS.
use super::statum_codec::{StratumCommand, StratumLine, StratumLinePayload};
use crate::Error;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

/// Upper bound on a decoded prompt / answer carried over stratum (upstream value).
pub(super) const MAX_PAYLOAD_BYTES: usize = 1_048_576;

#[derive(Debug)]
pub(super) struct AiRequest {
    pub task_id: String,
    pub request_hash: String,
    pub model_hex: String,
    pub model_id: [u8; 32],
    pub prompt: String,
    pub max_tokens: usize,
}

impl AiRequest {
    /// `params = [task_id, txid, request_hash, model_id_hex, prompt_base64, max_tokens, reward]`;
    /// task_id must equal request_hash (case-insensitive); ids are 64 hex chars.
    pub fn parse(fields: (String, String, String, String, String, u32, String)) -> Result<Self, Error> {
        let (task_id, txid, request_hash, model_hex, prompt_b64, max_tokens, reward) = fields;
        if !task_id.eq_ignore_ascii_case(&request_hash) || max_tokens == 0 || max_tokens > i32::MAX as u32 {
            return Err("Invalid AI task metadata".into());
        }
        for value in [&task_id, &txid, &request_hash, &model_hex] {
            if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid AI task identifier".into());
            }
        }
        reward.parse::<u64>()?;
        if prompt_b64.len() > MAX_PAYLOAD_BYTES.div_ceil(3) * 4 {
            return Err("AI prompt is too large".into());
        }
        let prompt = String::from_utf8(BASE64.decode(prompt_b64)?)?;
        if prompt.len() > MAX_PAYLOAD_BYTES || prompt.contains('\0') {
            return Err("Invalid AI prompt".into());
        }
        let mut model_id = [0; 32];
        hex::decode_to_slice(&model_hex, &mut model_id)?;
        Ok(Self { task_id, request_hash, model_hex, model_id, prompt, max_tokens: max_tokens as usize })
    }

    /// `mining.ai_response` with `params = [worker, task_id, request_hash, model_id_hex,
    /// answer_base64]`. An empty answer is never sent (the pool re-dispatches on silence).
    pub fn response(&self, id: u32, worker: String, result: &str) -> Result<StratumLine, Error> {
        if result.is_empty() || result.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI result size".into());
        }
        Ok(StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiResponse((
                worker,
                self.task_id.clone(),
                self.request_hash.clone(),
                self.model_hex.clone(),
                BASE64.encode(result.as_bytes()),
            ))),
            jsonrpc: Some("2.0".into()),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(prompt: &str) -> (String, String, String, String, String, u32, String) {
        let h = "ab".repeat(32);
        (h.clone(), "cd".repeat(32), h.to_uppercase(), "73740b443bdc00afda5fa34eb9999d3fea77dcc3f6de238fab701394cdc96fb3".into(), BASE64.encode(prompt), 64, "200000000".into())
    }

    #[test]
    fn parses_a_pool_request_and_builds_the_response() {
        let req = AiRequest::parse(fields("Hello")).unwrap();
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_tokens, 64);
        assert_eq!(hex::encode(req.model_id), "73740b443bdc00afda5fa34eb9999d3fea77dcc3f6de238fab701394cdc96fb3");
        let line = req.response(7, "worker".into(), "Hi there").unwrap();
        let json = serde_json::to_value(&line).unwrap();
        assert_eq!(json["method"], "mining.ai_response");
        assert_eq!(json["id"], 7);
        assert_eq!(json["params"][0], "worker");
        assert_eq!(json["params"][1], "ab".repeat(32));
        assert_eq!(json["params"][4], BASE64.encode("Hi there"));
        assert!(req.response(8, "w".into(), "").is_err(), "never send an empty answer");
    }

    #[test]
    fn rejects_malformed_requests() {
        let mut f = fields("x");
        f.2 = "ef".repeat(32); // task_id != request_hash
        assert!(AiRequest::parse(f).is_err());
        let mut f = fields("x");
        f.5 = 0;
        assert!(AiRequest::parse(f).is_err());
        let mut f = fields("x");
        f.3 = "zz".repeat(32);
        assert!(AiRequest::parse(f).is_err());
        let mut f = fields("x");
        f.6 = "-1".into();
        assert!(AiRequest::parse(f).is_err());
        assert!(AiRequest::parse(fields("nul\0byte")).is_err());
        let mut f = fields("x");
        f.4 = "%%%".into();
        assert!(AiRequest::parse(f).is_err());
    }

    #[test]
    fn decodes_the_wire_line() {
        let raw = format!(
            r#"{{"id":null,"method":"mining.ai_request","params":["{h}","{t}","{h}","{m}","{p}",128,"5"]}}"#,
            h = "11".repeat(32),
            t = "22".repeat(32),
            m = "33".repeat(32),
            p = BASE64.encode("prompt")
        );
        let line: StratumLine = serde_json::from_str(&raw).unwrap();
        match line.payload {
            StratumLinePayload::StratumCommand(StratumCommand::MiningAiRequest(f)) => {
                assert_eq!(AiRequest::parse(f).unwrap().prompt, "prompt");
            }
            other => panic!("decoded as {other:?}"),
        }
    }
}

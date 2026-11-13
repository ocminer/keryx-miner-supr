//! `mining.ai_request` / `mining.ai_response` — pool-dispatched AI inference (upstream
//! keryx-miner 2cf7ab5, neuropool protocol). From the H14 gate every on-chain request is a
//! private envelope sealed to the tier cohort's escrow keys; in pool mining that cohort member is
//! the POOL (its escrow key signs the coinbase), so a v3 pool opens the envelope itself, sends the
//! plaintext prompt here, and seals + signs the answer it gets back. The miner never sees a key
//! and nothing is published to IPFS.
use super::statum_codec::{ErrorCode, StratumCommand, StratumLine, StratumLinePayload};
use crate::Error;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

/// Upper bound on a decoded prompt / answer carried over stratum (upstream value).
pub(super) const MAX_PAYLOAD_BYTES: usize = 1_048_576;

/// The H15 trace a pool asks for (0.15.3 optional 8th `mining.ai_request` param).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TraceAsk {
    pub tier: u8,
    pub prompt_len: u32,
    pub chain: bool,
    pub attention: bool,
    /// Where to POST the `KRXREC01` compact record (pool custody, option 3); `None`: keep it local.
    pub upload: Option<String>,
}

impl TraceAsk {
    fn parse(ext: &serde_json::Value, sent_len: usize) -> Result<Option<Self>, Error> {
        let Some(t) = ext.get("trace") else { return Ok(None) };
        if t.is_null() {
            return Ok(None);
        }
        let tier = t.get("tier").and_then(|v| v.as_u64()).filter(|v| *v < 16).ok_or("trace.tier missing")? as u8;
        // Upstream: omitted metadata defaults to the ORIGINAL opened bytes, never the inflated
        // text. Clamp both forms to the consensus private-payload bound.
        let limit = keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN as u32;
        let prompt_len = match t.get("prompt_len") {
            None => sent_len.min(limit as usize) as u32,
            Some(v) => v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or("Invalid AI trace prompt length")?.min(limit),
        };
        let chain = t.get("chain").and_then(|v| v.as_bool()).unwrap_or(true);
        let attention = t.get("attention").and_then(|v| v.as_bool()).unwrap_or(true);
        let upload = match t.get("upload").and_then(|v| v.as_str()) {
            Some(u) if u.starts_with("https://") || u.starts_with("http://") => Some(u.to_string()),
            Some(_) => return Err("trace.upload must be an http(s) URL".into()),
            None => None,
        };
        Ok(Some(Self { tier, prompt_len, chain, attention: chain && attention, upload }))
    }
}

#[derive(Debug)]
pub(super) struct AiRequest {
    pub task_id: String,
    pub request_hash: String,
    pub model_hex: String,
    pub model_id: [u8; 32],
    pub prompt: String,
    pub max_tokens: usize,
    /// H15: answer traced (pool seals `body`, attaches `trace` BEFORE signing).
    pub trace: Option<TraceAsk>,
}

impl AiRequest {
    /// `params = [task_id, txid, request_hash, model_id_hex, prompt_base64, max_tokens, reward]`;
    /// task_id must equal request_hash (case-insensitive); ids are 64 hex chars.
    pub fn parse(fields: (String, String, String, String, String, u32, String)) -> Result<Self, Error> {
        Self::parse_ext(fields, None)
    }

    /// [`parse`] with the optional 0.15.3 extension object.
    pub fn parse_ext(fields: (String, String, String, String, String, u32, String), ext: Option<&serde_json::Value>) -> Result<Self, Error> {
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
        let raw = BASE64.decode(prompt_b64)?;
        if raw.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI prompt".into());
        }
        let trace = match ext {
            Some(e) => TraceAsk::parse(e, raw.len())?,
            None => None,
        };
        // Since H14 an opened private prompt may be COMPRESSED (magic 00 4B 5A 01 + raw DEFLATE): a
        // pool forwards the opened bytes as they are, so inflate them exactly like the solo (gRPC)
        // path and upstream do (cut at 1 MiB; a broken stream yields its partial output, else the
        // raw bytes; lossy UTF-8). The trace keeps the pool's `prompt_len`, original opened length the
        // node sees, while the engine tokenizes the inflated text.
        let prompt = crate::client::grpc::decode_private_prompt(&raw);
        // The node accepts NUL bytes in a prompt. A traced (H15 network) answer tokenizes the full
        // prompt bytes through the length-aware engine entry point, so it is served. An untraced
        // request (pool chat/probe, pre-H15) goes through llama.cpp's C-string chat template,
        // which would silently truncate at the NUL: refuse that one explicitly instead.
        if trace.is_none() && prompt.contains('\0') {
            return Err("AI prompt contains a NUL byte and the request is untraced (only traced H15 requests can carry one)".into());
        }
        let mut model_id = [0; 32];
        hex::decode_to_slice(&model_hex, &mut model_id)?;
        Ok(Self { task_id, request_hash, model_hex, model_id, prompt, max_tokens: max_tokens as usize, trace })
    }

    /// The request hash as bytes (validated hex by [`parse_ext`]).
    pub fn request_hash_bytes(&self) -> [u8; 32] {
        let mut h = [0u8; 32];
        let _ = hex::decode_to_slice(&self.request_hash, &mut h);
        h
    }

    /// Upstream refuses an unservable dispatch with error 46, if it has an id. The pool can
    /// immediately re-dispatch; this is not a mining share response or an empty AI answer.
    pub fn refusal(&self, reply_id: Option<u32>, message: &str) -> Option<StratumLine> {
        reply_id.map(|id| StratumLine::error_reply(id, ErrorCode::Busy, message))
    }

    /// `mining.ai_response` with `params = [worker, task_id, request_hash, model_id_hex,
    /// answer_base64]`. An empty answer is never sent; refusals use the original dispatch id.
    pub fn response(&self, id: u32, worker: String, result: &str) -> Result<StratumLine, Error> {
        self.response_traced(id, worker, result, None)
    }

    /// The response of a traced answer: the 5 positional params plus `{"trace", "body"}` (hex).
    pub fn response_traced(&self, id: u32, worker: String, result: &str, traced: Option<(&[u8], &[u8])>) -> Result<StratumLine, Error> {
        if result.is_empty() || result.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI result size".into());
        }
        let mut params: Vec<serde_json::Value> = vec![
            worker.into(),
            self.task_id.clone().into(),
            self.request_hash.clone().into(),
            self.model_hex.clone().into(),
            BASE64.encode(result.as_bytes()).into(),
        ];
        if let Some((commitment, body)) = traced {
            params.push(serde_json::json!({ "trace": hex::encode(commitment), "body": hex::encode(body) }));
        }
        Ok(StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiResponse(params)),
            jsonrpc: Some("2.0".into()),
            error: None,
        })
    }
}

/// A pool's `mining.ai_opening`: build the opening (or attention phase) of a traced answer.
#[derive(Debug)]
pub(super) struct AiOpening {
    pub request_hash: String,
    pub response_txid: [u8; 32],
    pub response_txid_hex: String,
    pub seed: [u8; 32],
    pub path: Vec<u64>,
    pub prev: Vec<u8>,
    /// Option 3 fallback: where to GET the `KRXREC01` record when this rig did not answer itself.
    pub record_url: Option<String>,
}

impl AiOpening {
    pub fn parse(v: &[serde_json::Value]) -> Result<Self, Error> {
        if v.len() < 3 {
            return Err("mining.ai_opening takes [request_hash, response_txid, seed, path, prev]".into());
        }
        let hex32 = |i: usize| -> Result<([u8; 32], String), Error> {
            let s = v[i].as_str().ok_or("mining.ai_opening id must be a string")?;
            let mut b = [0u8; 32];
            hex::decode_to_slice(s, &mut b)?;
            Ok((b, s.to_ascii_lowercase()))
        };
        let (_, request_hash) = hex32(0)?;
        let (response_txid, response_txid_hex) = hex32(1)?;
        let (seed, _) = hex32(2)?;
        let path = match v.get(3) {
            Some(serde_json::Value::Array(a)) => a.iter().map(|x| x.as_u64().ok_or("attention path must be u64s")).collect::<Result<Vec<_>, _>>()?,
            None | Some(serde_json::Value::Null) => Vec::new(),
            _ => return Err("attention path must be an array".into()),
        };
        let prev = match v.get(4).and_then(|x| x.as_str()) {
            Some(s) if !s.is_empty() => hex::decode(s)?,
            _ => Vec::new(),
        };
        if path.is_empty() != prev.is_empty() {
            return Err("an attention phase needs both its path and the previous partials".into());
        }
        let record_url = match v.get(5).and_then(|x| x.as_str()) {
            Some(u) if u.starts_with("https://") || u.starts_with("http://") => Some(u.to_string()),
            Some("") | None => None,
            Some(_) => return Err("record url must be http(s)".into()),
        };
        Ok(Self { request_hash, response_txid, response_txid_hex, seed, path, prev, record_url })
    }

    /// `mining.ai_opening_result` [worker, request_hash, response_txid, path, body_hex, error].
    pub fn result(&self, id: u32, worker: String, body: Result<&[u8], &str>) -> StratumLine {
        let (body_hex, err) = match body {
            Ok(b) => (hex::encode(b), String::new()),
            Err(e) => (String::new(), e.to_string()),
        };
        StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiOpeningResult(vec![
                worker.into(),
                self.request_hash.clone().into(),
                self.response_txid_hex.clone().into(),
                serde_json::json!(self.path),
                body_hex.into(),
                err.into(),
            ])),
            jsonrpc: Some("2.0".into()),
            error: None,
        }
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
        assert!(AiRequest::parse(fields("nul\0byte")).is_err(), "untraced: no C-string-safe path");
        let mut f = fields("x");
        f.4 = "%%%".into();
        assert!(AiRequest::parse(f).is_err());
    }

    #[test]
    fn traced_request_keeps_nul_bytes_in_the_prompt() {
        let prompt = "before\0after\0";
        let ext = serde_json::json!({"trace": {"tier": 0, "prompt_len": prompt.len(), "chain": true, "attention": true}});
        let req = AiRequest::parse_ext(fields(prompt), Some(&ext)).unwrap();
        assert_eq!(req.prompt.as_bytes(), prompt.as_bytes(), "never stripped or replaced");
        assert_eq!(req.prompt.len(), 13);
        assert!(req.trace.is_some());
    }

    fn compressed(text: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(text).unwrap();
        let mut packed = vec![0x00, b'K', b'Z', 0x01];
        packed.extend(enc.finish().unwrap());
        packed
    }

    fn fields_bytes(prompt: &[u8]) -> (String, String, String, String, String, u32, String) {
        let mut f = fields("x");
        f.4 = BASE64.encode(prompt);
        f
    }

    #[test]
    fn parse_inflates_a_compressed_private_prompt() {
        let text = "Entrée 41 — le code est ORCHIDÉE-7319. ".repeat(500);
        let packed = compressed(text.as_bytes());
        assert!(packed.len() < text.len() / 10);
        assert_eq!(AiRequest::parse(fields_bytes(&packed)).unwrap().prompt, text);
        // Traced: the engine gets the inflated text, the trace keeps the pool's (sealed) length.
        let ext = serde_json::json!({"trace": {"tier": 0, "prompt_len": packed.len(), "chain": true, "attention": true}});
        let req = AiRequest::parse_ext(fields_bytes(&packed), Some(&ext)).unwrap();
        assert_eq!(req.prompt, text);
        assert_eq!(req.trace.unwrap().prompt_len as usize, packed.len());
    }

    #[test]
    fn parse_serves_a_truncated_compressed_prompt_as_its_partial_output() {
        let ascii: String = (0..4000).map(|i| char::from(b'a' + (i * 7 % 26) as u8)).collect();
        let mut packed = compressed(ascii.as_bytes());
        packed.truncate(packed.len() / 2);
        let prompt = AiRequest::parse(fields_bytes(&packed)).unwrap().prompt;
        assert!(!prompt.is_empty() && ascii.starts_with(&prompt), "upstream: partial output");
    }

    #[test]
    fn parse_serves_a_corrupt_compressed_prompt_as_its_raw_bytes() {
        let mut corrupt = vec![0x00, b'K', b'Z', 0x01];
        corrupt.extend([0xFF; 32]);
        assert_eq!(AiRequest::parse(fields_bytes(&corrupt)).unwrap().prompt, "\u{FFFD}".repeat(32), "upstream: raw bytes, lossy");
    }

    #[test]
    fn parse_caps_the_raw_and_the_inflated_prompt() {
        assert!(AiRequest::parse(fields_bytes(&vec![b'a'; MAX_PAYLOAD_BYTES + 1])).is_err(), "raw length checked");
        let bomb = compressed(&vec![b'a'; MAX_PAYLOAD_BYTES + 4096]);
        assert!(bomb.len() < MAX_PAYLOAD_BYTES);
        assert_eq!(AiRequest::parse(fields_bytes(&bomb)).unwrap().prompt.len(), 1 << 20, "inflated output cut at 1 MiB");
    }

    #[test]
    fn parse_reads_invalid_utf8_lossily_and_keeps_nul_rules_after_inflating() {
        assert_eq!(AiRequest::parse(fields_bytes(&[b'h', b'i', 0xFF])).unwrap().prompt, "hi\u{FFFD}");
        let nul = compressed(b"before\0after");
        assert!(AiRequest::parse(fields_bytes(&nul)).is_err(), "untraced NUL refused after inflating too");
        let ext = serde_json::json!({"trace": {"tier": 0, "prompt_len": nul.len(), "chain": true, "attention": true}});
        assert_eq!(AiRequest::parse_ext(fields_bytes(&nul), Some(&ext)).unwrap().prompt.as_bytes(), b"before\0after");
    }

    #[test]
    fn large_compressed_requests_keep_original_trace_length_and_pass_admission() {
        for size in [74_739, 79_150, 70 * 1024, 100 * 1024, 250 * 1024, 500 * 1024, 1 << 20] {
            let text = "x".repeat(size);
            let packed = compressed(text.as_bytes());
            let ext = serde_json::json!({"trace": {"tier": 3}});
            let req = AiRequest::parse_ext(fields_bytes(&packed), Some(&ext)).unwrap();
            assert_eq!(req.prompt, text);
            assert_eq!(req.trace.unwrap().prompt_len as usize, packed.len(), "default is before inflate");
            assert!(keryx_miner::slm::validate_private_inference_request(&req.prompt, 128, 0).is_ok());
            let explicit = serde_json::json!({"trace": {"tier": 3, "prompt_len": 23_000}});
            let req = AiRequest::parse_ext(fields_bytes(&packed), Some(&explicit)).unwrap();
            assert_eq!(req.trace.unwrap().prompt_len, 23_000, "explicit node metadata is authoritative");
        }
    }

    #[test]
    fn trace_length_defaults_clamps_and_rejects_bad_metadata_like_upstream() {
        let ext = serde_json::json!({"trace": {"tier": 0, "chain": false, "attention": true}});
        let req = AiRequest::parse_ext(fields(&"x".repeat(70_000)), Some(&ext)).unwrap();
        let trace = req.trace.unwrap();
        assert_eq!(trace.prompt_len, 65_536);
        assert!(!trace.attention, "attention requires chain");
        for length in [0u64, 17, 65_536, 100_000, u32::MAX as u64] {
            let ext = serde_json::json!({"trace": {"tier": 0, "prompt_len": length}});
            assert_eq!(AiRequest::parse_ext(fields("small"), Some(&ext)).unwrap().trace.unwrap().prompt_len, length.min(65_536) as u32);
        }
        for length in [serde_json::Value::Null, serde_json::json!(-1), serde_json::json!(1u64 << 32), serde_json::json!("7"), serde_json::json!(1.5)] {
            let ext = serde_json::json!({"trace": {"tier": 0, "prompt_len": length}});
            assert!(AiRequest::parse_ext(fields("small"), Some(&ext)).is_err());
        }
    }

    #[test]
    fn upstream_byte_cap_precedes_lossy_utf8_expansion() {
        let ext = serde_json::json!({"trace": {"tier": 0}});
        let packed = compressed(&vec![0xFF; MAX_PAYLOAD_BYTES + 4096]);
        let req = AiRequest::parse_ext(fields_bytes(&packed), Some(&ext)).unwrap();
        assert_eq!(req.prompt, "\u{FFFD}".repeat(MAX_PAYLOAD_BYTES));
        assert!(keryx_miner::slm::validate_private_inference_request(&req.prompt, 128, 0).is_ok());
        // A valid multibyte prompt may end in a partial codepoint at the upstream byte cap.
        let text = format!("xxx{}", "🙂".repeat(MAX_PAYLOAD_BYTES / 4));
        let packed = compressed(text.as_bytes());
        let req = AiRequest::parse_ext(fields_bytes(&packed), Some(&ext)).unwrap();
        assert_eq!(req.prompt.len(), MAX_PAYLOAD_BYTES + 2);
        assert!(req.prompt.ends_with('\u{FFFD}'));
        assert!(keryx_miner::slm::validate_private_inference_request(&req.prompt, 128, 0).is_ok());
    }

    #[test]
    fn refusal_uses_dispatch_id_and_upstream_busy_error_not_an_answer() {
        let req = AiRequest::parse(fields("hello")).unwrap();
        let wire = serde_json::to_value(req.refusal(Some(991), "model not ready").unwrap()).unwrap();
        assert_eq!(wire["id"], 991);
        assert!(wire["result"].is_null());
        assert_eq!(wire["error"][0], 46);
        assert_eq!(wire["error"][1], "model not ready");
        assert!(wire.get("method").is_none());
        assert!(req.refusal(None, "busy").is_none());
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
                assert!(f.ext.is_none());
                assert_eq!(AiRequest::parse(f.fields).unwrap().prompt, "prompt");
            }
            other => panic!("decoded as {other:?}"),
        }
    }

    #[test]
    fn traced_requests_and_openings_round_trip() {
        let raw = format!(
            r#"{{"id":null,"method":"mining.ai_request","params":["{h}","{t}","{h}","{m}","{p}",128,"5",{{"trace":{{"tier":2,"prompt_len":17,"chain":true,"attention":true}}}}]}}"#,
            h = "11".repeat(32),
            t = "22".repeat(32),
            m = "33".repeat(32),
            p = BASE64.encode("prompt")
        );
        let line: StratumLine = serde_json::from_str(&raw).unwrap();
        let StratumLinePayload::StratumCommand(StratumCommand::MiningAiRequest(f)) = line.payload else { panic!("not an ai_request") };
        let req = AiRequest::parse_ext(f.fields, f.ext.as_ref()).unwrap();
        assert_eq!(req.trace, Some(TraceAsk { tier: 2, prompt_len: 17, chain: true, attention: true, upload: None }));
        assert_eq!(req.request_hash_bytes(), [0x11; 32]);
        let out = req.response_traced(9, "w".into(), "answer", Some((&[1u8, 2], &[3u8]))).unwrap();
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["params"].as_array().unwrap().len(), 6);
        assert_eq!(json["params"][5]["trace"], "0102");
        assert_eq!(json["params"][5]["body"], "03");
        // An untraced answer keeps the historical 5 params.
        assert_eq!(serde_json::to_value(req.response(10, "w".into(), "a").unwrap()).unwrap()["params"].as_array().unwrap().len(), 5);

        let o = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [2, 0], "beef"]);
        let op = AiOpening::parse(o.as_array().unwrap()).unwrap();
        assert_eq!((op.path.clone(), op.prev.clone(), op.seed), (vec![2, 0], vec![0xbe, 0xef], [0xbb; 32]));
        let first = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [], ""]);
        assert!(AiOpening::parse(first.as_array().unwrap()).unwrap().path.is_empty());
        let bad = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [1], ""]);
        assert!(AiOpening::parse(bad.as_array().unwrap()).is_err());
        let r = serde_json::to_value(&op.result(3, "w".into(), Ok(&[0xb0, 1]))).unwrap();
        assert_eq!(r["method"], "mining.ai_opening_result");
        assert_eq!(r["params"][4], "b001");
    }
}

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

/// The H15 trace a pool asks for (0.15.3 optional 8th `mining.ai_request` param).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TraceAsk {
    pub tier: u8,
    pub prompt_len: u32,
    pub chain: bool,
    pub attention: bool,
}

impl TraceAsk {
    fn parse(ext: &serde_json::Value) -> Result<Option<Self>, Error> {
        let Some(t) = ext.get("trace") else { return Ok(None) };
        if t.is_null() {
            return Ok(None);
        }
        let tier = t.get("tier").and_then(|v| v.as_u64()).filter(|v| *v < 16).ok_or("trace.tier missing")? as u8;
        let prompt_len = t.get("prompt_len").and_then(|v| v.as_u64()).and_then(|v| u32::try_from(v).ok()).ok_or("trace.prompt_len missing")?;
        let chain = t.get("chain").and_then(|v| v.as_bool()).unwrap_or(true);
        let attention = t.get("attention").and_then(|v| v.as_bool()).unwrap_or(true);
        Ok(Some(Self { tier, prompt_len, chain, attention }))
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
        let trace = match ext {
            Some(e) => TraceAsk::parse(e)?,
            None => None,
        };
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
        Ok(Self { task_id, request_hash, model_hex, model_id, prompt, max_tokens: max_tokens as usize, trace })
    }

    /// The request hash as bytes (validated hex by [`parse_ext`]).
    pub fn request_hash_bytes(&self) -> [u8; 32] {
        let mut h = [0u8; 32];
        let _ = hex::decode_to_slice(&self.request_hash, &mut h);
        h
    }

    /// `mining.ai_response` with `params = [worker, task_id, request_hash, model_id_hex,
    /// answer_base64]`. An empty answer is never sent (the pool re-dispatches on silence).
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
        Ok(Self { request_hash, response_txid, response_txid_hex, seed, path, prev })
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
        assert_eq!(req.trace, Some(TraceAsk { tier: 2, prompt_len: 17, chain: true, attention: true }));
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

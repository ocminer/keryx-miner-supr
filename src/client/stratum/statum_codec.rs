use bytes::BytesMut;
use log::error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_repr::*;
use std::fmt::{Display, Formatter};
use std::{fmt, io};
use tokio_util::codec::{Decoder, Encoder, LinesCodec};

#[derive(Serialize_repr, Deserialize_repr, Debug, Clone)]
#[repr(u8)]
pub enum ErrorCode {
    Unknown = 20,
    JobNotFound = 21,
    DuplicateShare = 22,
    LowDifficultyShare = 23,
    Unauthorized = 24,
    NotSubscribed = 25,
}

impl Display for ErrorCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self {
            ErrorCode::Unknown => write!(f, "Unknown"),
            ErrorCode::JobNotFound => write!(f, "JobNotFound"),
            ErrorCode::DuplicateShare => write!(f, "DuplicateShare"),
            ErrorCode::LowDifficultyShare => write!(f, "LowDifficultyShare"),
            ErrorCode::Unauthorized => write!(f, "Unauthorized"),
            ErrorCode::NotSubscribed => write!(f, "NotSubscribed"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct StratumError(pub(crate) ErrorCode, pub(crate) String, #[serde(default)] pub(crate) Option<Value>);

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum MiningNotify {
    // 5-element: job_id, header_hash, timestamp, daa_score, task (AiRequest payload).
    // The pool sends the task as a JSON OBJECT directly (the locked H6 contract), so this is a
    // `Value`; a legacy double-encoded JSON string still deserializes as `Value::String` and is
    // unwrapped at the call site. Typing this as `String` (the old assumption) made the whole
    // `mining.notify` line fail untagged deserialization when the pool sent an object.
    MiningNotifyWithTask((String, [u64; 4], u64, u64, Value)),
    MiningNotifyShortV2((String, [u64; 4], u64, u64)),
    MiningNotifyShort((String, [u64; 4], u64)),
    MiningNotifyLong((String, String, String, String, Vec<String>, String, String, String, bool)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubmit {
    // 8-element (PoM + H6 UNSIGNED AiResponse, service-bond era): address, job_id, nonce, opoi_tag,
    // ipfs_cid, pom_proof_hex, reqId, response_length (token count = answer.split_whitespace().count()).
    // Sent at/after the PoM v3 gate when a CID (an answer) is present. The POOL — not the miner —
    // builds the coinbase and therefore signs the V2 AiResponse with the POOL's escrow key over the
    // exact 78 v1 bytes; `response_length` MUST be transmitted (not re-derived) because it is one of
    // those signed bytes, and `reqId` echoes the dispatched request so the pool can match it. The
    // trailing u32 makes this arity-8 shape distinct from any all-string tuple; listed first so
    // untagged deserialization never mistakes it for the 6-element PoM submit.
    MiningSubmitWithUnsignedResponse((String, String, String, String, String, String, String, u32)),
    // 6-element (PoM, post-fork): address, job_id, nonce, opoi_tag, ipfs_cid (or ""),
    // pom_proof_hex. Fixed slot layout — CID stays at params[4] even when empty so the
    // proof is always params[5]. See POM_STRATUM_RECIPE.md (pool side reconciled).
    MiningSubmitWithPom((String, String, String, String, String, String)),
    // 5-element: address, job_id, nonce, opoi_tag, ipfs_cid (Phase 2 full inference submit)
    MiningSubmitWithCID((String, String, String, String, String)),
    MiningSubmitWithTag((String, String, String, String)), // address, job_id, nonce, opoi_tag
    MiningSubmitShort((String, String, String)),
}

/// Interactive chat inference request (pool → miner, `mining.inference_request`). Off-chain
/// product path — NOT the consensus AiResponse flow: no tx, no escrow, text returned inline. The
/// `params` value is this object directly (not wrapped in an array).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct InferenceRequestParams {
    #[serde(rename = "reqId")]
    pub req_id: String,
    /// 64-hex tier model id the chat should be answered by.
    pub model_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    /// Wall-clock budget the pool allots for routing this chat to a free card (ms). 0/absent ⇒ the
    /// router's default (30s). If every eligible card stays busy past it, the miner replies busy.
    #[serde(default)]
    pub deadline_ms: u64,
}

/// Interactive chat inference result (miner → pool, `mining.inference_result`). On success:
/// `{ reqId, ok:true, text, tokens, ms }`; on failure/timeout/unready-model: `{ reqId, ok:false,
/// error }`. Optional fields are omitted when absent so both shapes serialize cleanly.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct InferenceResultParams {
    #[serde(rename = "reqId")]
    pub req_id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubscribe {
    MiningSubscribeDefault((String,)),
    MiningSubscribeOptions((String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SetExtranonce {
    SetExtranoncePlain((String, u32)),
    SetExtranoncePlainEth((String,)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "method", content = "params")]
pub(crate) enum StratumCommand {
    #[serde(rename = "mining.set_extranonce", alias = "set_extranonce")]
    SetExtranonce(SetExtranonce),
    #[serde(rename = "mining.set_difficulty")]
    MiningSetDifficulty((f32,)),
    #[serde(rename = "mining.notify")]
    MiningNotify(MiningNotify),
    #[serde(rename = "mining.subscribe")]
    Subscribe(MiningSubscribe),
    #[serde(rename = "mining.authorize")]
    Authorize((String, String)),
    #[serde(rename = "mining.submit")]
    MiningSubmit(MiningSubmit),
    // Phase 2 OPoI: miner → bridge — declare loaded SLM model IDs (sent after authorize)
    #[serde(rename = "mining.declare_capabilities")]
    MiningDeclareCapabilities(Vec<String>),
    // Telemetry (v0.7.0): miner → pool — STATIC rig identity, sent once after subscribe. params =
    // [ hello-object ]. DISPLAY/OPS ONLY (spoofable). Best-effort; pool may reply error 20.
    #[serde(rename = "mining.hello")]
    MiningHello((Value,)),
    // Telemetry (v0.7.0): miner → pool — DYNAMIC per-GPU metrics, sent periodically. params =
    // [ metrics-object ]. DISPLAY/OPS ONLY (spoofable).
    #[serde(rename = "mining.telemetry")]
    MiningTelemetry((Value,)),
    // Phase 2 OPoI: bridge → miner — "model_id_hex:nonce_hex" capability challenge
    #[serde(rename = "mining.challenge")]
    MiningChallenge((String, String)),
    // Phase 2 OPoI: miner → bridge — [model_id_hex, nonce_hex, result_text] challenge
    // response. The nonce is echoed back so the bridge can reject replayed/stale responses.
    #[serde(rename = "mining.challenge_response")]
    MiningChallengeResponse((String, String, String)),
    // H6 interactive chat (protocol extension): pool → miner — off-chain, low-latency inference
    // request. `params` is the object itself (reqId/model_id/prompt/max_tokens/stream). Separate
    // from consensus AiResponse; the miner answers inline via `mining.inference_result`.
    #[serde(rename = "mining.inference_request")]
    MiningInferenceRequest(InferenceRequestParams),
    // H6 interactive chat (protocol extension): miner → pool — the chat answer (text inline) or an
    // error. `params` is the object itself (reqId/ok/text/tokens/ms | reqId/ok/error).
    #[serde(rename = "mining.inference_result")]
    MiningInferenceResult(InferenceResultParams),
    /*#[serde(rename = "mining.submit_hashrate")]
    MiningSubmitHashrate {
        params: (String, String),
        worker: String,
    },*/ //{"id":9,"method":"mining.submit_hashrate","jsonrpc":"2.0","worker":"rig","params":["0x00000000000000000000000000000000","0x85198cd10b915d560722cdfdf490d4d93892d2cc3fa5f2ff2195d499d04ee54c"]}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumResult {
    Plain(Option<bool>),
    Eth((bool, String)),
    Subscribe((Vec<(String, String)>, String, u32)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumLinePayload {
    StratumCommand(StratumCommand),
    StratumResult { result: StratumResult },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct StratumLine {
    pub(crate) id: Option<u32>,
    #[serde(flatten)]
    pub(crate) payload: StratumLinePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) jsonrpc: Option<String>,
    pub(crate) error: Option<StratumError>,
}

/// An error occurred while encoding or decoding a line.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum NewLineJsonCodecError {
    JsonParseError(String),
    JsonEncodeError,
    LineSplitError,
    LineEncodeError,
    Io(io::Error),
}

impl fmt::Display for NewLineJsonCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Some error occured")
    }
}
impl From<io::Error> for NewLineJsonCodecError {
    fn from(e: io::Error) -> NewLineJsonCodecError {
        NewLineJsonCodecError::Io(e)
    }
}
impl std::error::Error for NewLineJsonCodecError {}

impl From<(String, String)> for NewLineJsonCodecError {
    fn from(e: (String, String)) -> Self {
        NewLineJsonCodecError::JsonParseError(format!("{}: {}", e.0, e.1))
    }
}

pub(crate) struct NewLineJsonCodec {
    lines_codec: LinesCodec,
}

impl NewLineJsonCodec {
    pub fn new() -> Self {
        Self { lines_codec: LinesCodec::new() }
    }
}

impl Decoder for NewLineJsonCodec {
    type Item = StratumLine;
    type Error = NewLineJsonCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode(src) {
            Ok(Some(s)) => {
                serde_json::from_str::<StratumLine>(s.as_str()).map_err(|e| (e.to_string(), s).into()).map(Some)
            }
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode_eof(buf) {
            Ok(Some(s)) => serde_json::from_str(s.as_str()).map_err(|e| (e.to_string(), s).into()),
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
        }
    }
}

impl Encoder<StratumLine> for NewLineJsonCodec {
    type Error = NewLineJsonCodecError;

    fn encode(&mut self, item: StratumLine, dst: &mut BytesMut) -> Result<(), Self::Error> {
        match serde_json::to_string(&item) {
            Ok(json) => self.lines_codec.encode(json, dst).map_err(|_| NewLineJsonCodecError::LineEncodeError),
            Err(e) => {
                error!("Error! {:?}", e);
                Err(NewLineJsonCodecError::JsonEncodeError)
            }
        }
    }
}

impl Default for NewLineJsonCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H6 unsigned AiResponse submit: the wire must be exactly
    /// [address, job_id, nonce, opoi_tag, ipfs_cid, pom_proof_hex, reqId, response_length]
    /// with response_length as a JSON number, and must round-trip back to the same variant.
    #[test]
    fn unsigned_response_submit_roundtrip() {
        let line = StratumLine {
            id: Some(7),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                MiningSubmit::MiningSubmitWithUnsignedResponse((
                    "keryx:qaddr".to_string(),
                    "job_500000".to_string(),
                    "00000000deadbeef".to_string(),
                    "opoitag".to_string(),
                    "QmCidV0".to_string(),
                    "aabbcc".to_string(),
                    "req-42".to_string(),
                    17u32,
                )),
            )),
            jsonrpc: None,
            error: None,
        };

        let json = serde_json::to_string(&line).unwrap();
        // Exact wire shape: params is an 8-element array, reqId at [6], response_length (number) at [7].
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["method"], "mining.submit");
        let params = v["params"].as_array().unwrap();
        assert_eq!(params.len(), 8);
        assert_eq!(params[0], "keryx:qaddr");
        assert_eq!(params[4], "QmCidV0");
        assert_eq!(params[5], "aabbcc");
        assert_eq!(params[6], "req-42");
        assert_eq!(params[7], 17); // JSON number, not string
        assert!(params[7].is_u64());

        // Round-trips back to the same variant with identical fields.
        let back: StratumLine = serde_json::from_str(&json).unwrap();
        match back.payload {
            StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                MiningSubmit::MiningSubmitWithUnsignedResponse((addr, job, nonce, tag, cid, proof, rid, len)),
            )) => {
                assert_eq!(addr, "keryx:qaddr");
                assert_eq!(job, "job_500000");
                assert_eq!(nonce, "00000000deadbeef");
                assert_eq!(tag, "opoitag");
                assert_eq!(cid, "QmCidV0");
                assert_eq!(proof, "aabbcc");
                assert_eq!(rid, "req-42");
                assert_eq!(len, 17u32);
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }

    /// The plain 6-element PoM submit must still decode as `MiningSubmitWithPom` (not mistaken for
    /// the new 8-element unsigned-response shape).
    #[test]
    fn plain_pom_submit_still_decodes() {
        let raw = r#"{"id":1,"method":"mining.submit","params":["addr","job","nonce","tag","","aabb"],"error":null}"#;
        let line: StratumLine = serde_json::from_str(raw).unwrap();
        match line.payload {
            StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                MiningSubmit::MiningSubmitWithPom(_),
            )) => {}
            other => panic!("expected MiningSubmitWithPom, got {:?}", other),
        }
    }

    /// `mining.inference_request` (pool → miner): params is the object itself; decodes the fields.
    #[test]
    fn inference_request_decode() {
        let raw = r#"{"id":null,"method":"mining.inference_request","params":{"reqId":"c-1","model_id":"aa","prompt":"hi","max_tokens":128,"stream":false},"error":null}"#;
        let line: StratumLine = serde_json::from_str(raw).unwrap();
        match line.payload {
            StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceRequest(p)) => {
                assert_eq!(p.req_id, "c-1");
                assert_eq!(p.model_id, "aa");
                assert_eq!(p.prompt, "hi");
                assert_eq!(p.max_tokens, 128);
                assert!(!p.stream);
            }
            other => panic!("expected MiningInferenceRequest, got {:?}", other),
        }
    }

    /// `mining.inference_result` (miner → pool): OK shape carries text/tokens/ms and no error;
    /// error shape carries error and omits text/tokens/ms. Both round-trip.
    #[test]
    fn inference_result_roundtrip() {
        // success
        let ok = StratumLine {
            id: None,
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(
                InferenceResultParams {
                    req_id: "c-1".to_string(),
                    ok: true,
                    text: Some("hello world".to_string()),
                    tokens: Some(2),
                    ms: Some(42),
                    error: None,
                },
            )),
            jsonrpc: None,
            error: None,
        };
        let json = serde_json::to_string(&ok).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["method"], "mining.inference_result");
        assert_eq!(v["params"]["reqId"], "c-1");
        assert_eq!(v["params"]["ok"], true);
        assert_eq!(v["params"]["text"], "hello world");
        assert_eq!(v["params"]["tokens"], 2);
        assert_eq!(v["params"]["ms"], 42);
        assert!(v["params"].get("error").is_none()); // omitted when None
        let back: StratumLine = serde_json::from_str(&json).unwrap();
        match back.payload {
            StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(p)) => {
                assert!(p.ok);
                assert_eq!(p.text.as_deref(), Some("hello world"));
                assert_eq!(p.tokens, Some(2));
                assert_eq!(p.ms, Some(42));
                assert!(p.error.is_none());
            }
            other => panic!("wrong variant: {:?}", other),
        }

        // failure
        let err = StratumLine {
            id: None,
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(
                InferenceResultParams {
                    req_id: "c-2".to_string(),
                    ok: false,
                    text: None,
                    tokens: None,
                    ms: None,
                    error: Some("model not ready".to_string()),
                },
            )),
            jsonrpc: None,
            error: None,
        };
        let json = serde_json::to_string(&err).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["params"]["ok"], false);
        assert_eq!(v["params"]["error"], "model not ready");
        assert!(v["params"].get("text").is_none()); // omitted when None
    }
}

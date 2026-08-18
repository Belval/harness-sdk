//! Shared type definitions: messages, content blocks, media, tool specs, and
//! streaming events. Ports the TypeScript SDK's `src/types/` and adjacent
//! `tools/types.ts` / `models/streaming.ts` modules.

pub mod interrupt;
pub mod media;
pub mod messages;
pub mod streaming;
pub mod tools;

/// Serde helper that (de)serializes a `Vec<u8>` as a base64 string.
///
/// The TypeScript SDK encodes raw byte fields (`Uint8Array`) as base64 strings
/// in `toJSON()` and decodes them back in `fromJSON()`; this mirrors that so the
/// serialized wire form matches across SDKs.
pub(crate) mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        serializer.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD
            .decode(encoded.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

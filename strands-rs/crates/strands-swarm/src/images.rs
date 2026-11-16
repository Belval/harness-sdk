//! Images attached to mail: the mailbox keeps their paths, and the bytes are read only when the
//! receiving agent takes the mail, as image blocks of its next model input.

use std::path::Path;

use strands_agents::types::{ContentBlock, ImageBlock, ImageFormat, ImageSource, ToolResultContent};

use crate::runtime::Mail;

/// Largest image sent to a model (the Bedrock limit for one image).
pub const MAX_IMAGE_BYTES: u64 = 3_750_000;

/// The model-side format of an image file, by extension.
pub fn format_of(path: &Path) -> Option<ImageFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some(ImageFormat::Png),
        "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
        "gif" => Some(ImageFormat::Gif),
        "webp" => Some(ImageFormat::Webp),
        _ => None,
    }
}

/// Check that `path` can be attached: a supported format, readable, at most [`MAX_IMAGE_BYTES`].
pub fn check(path: &Path) -> Result<(), String> {
    if format_of(path).is_none() {
        return Err(format!("{}: not a png, jpg, gif or webp image", path.display()));
    }
    let size = std::fs::metadata(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .len();
    if size > MAX_IMAGE_BYTES {
        return Err(format!(
            "{}: {size} bytes is over the {MAX_IMAGE_BYTES}-byte limit for one image",
            path.display()
        ));
    }
    Ok(())
}

/// The image block for the file at `path`.
pub fn load(path: &Path) -> Result<ImageBlock, String> {
    check(path)?;
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(ImageBlock {
        format: format_of(path).expect("checked"),
        source: ImageSource::Bytes(bytes),
    })
}

/// The images attached to `mail`, as content for the receiver; one that cannot be read anymore
/// becomes a note saying why.
fn attachments(mail: &[Mail]) -> Vec<Result<ImageBlock, String>> {
    mail.iter()
        .flat_map(|mail| &mail.images)
        .map(|path| load(path).map_err(|error| format!("[attached image unavailable: {error}]")))
        .collect()
}

/// A new turn's input blocks for `text` and the images attached to `mail`.
pub fn input_blocks(text: String, mail: &[Mail]) -> Vec<ContentBlock> {
    let mut blocks = vec![ContentBlock::Text(text)];
    for attachment in attachments(mail) {
        blocks.push(match attachment {
            Ok(image) => ContentBlock::Image(image),
            Err(note) => ContentBlock::Text(note),
        });
    }
    blocks
}

/// The images attached to `mail`, as tool-result content (for mail that arrives mid-turn).
pub fn result_content(mail: &[Mail]) -> Vec<ToolResultContent> {
    attachments(mail)
        .into_iter()
        .map(|attachment| match attachment {
            Ok(image) => ToolResultContent::Image(image),
            Err(note) => ToolResultContent::Text(note),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knows_image_formats_by_extension() {
        assert_eq!(format_of(Path::new("a/b.PNG")), Some(ImageFormat::Png));
        assert_eq!(format_of(Path::new("x.jpeg")), Some(ImageFormat::Jpeg));
        assert_eq!(format_of(Path::new("x.txt")), None);
        assert_eq!(format_of(Path::new("png")), None);
    }

    #[test]
    fn checks_format_existence_and_size() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("ok.png");
        std::fs::write(&ok, b"\x89PNG").unwrap();
        assert!(check(&ok).is_ok());
        assert!(check(&dir.path().join("missing.png"))
            .unwrap_err()
            .contains("missing.png"));
        assert!(check(&dir.path().join("notes.txt"))
            .unwrap_err()
            .contains("not a png"));
        let big = dir.path().join("big.png");
        std::fs::write(&big, vec![0u8; MAX_IMAGE_BYTES as usize + 1]).unwrap();
        assert!(check(&big).unwrap_err().contains("limit"));
    }

    #[test]
    fn attachments_become_image_blocks_or_notes() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("ok.png");
        std::fs::write(&ok, b"\x89PNG").unwrap();
        let mail = Mail {
            from: "user".into(),
            from_label: "user".into(),
            kind: "message",
            text: "look".into(),
            images: vec![ok, dir.path().join("gone.png")],
        };
        let blocks = input_blocks("look".into(), std::slice::from_ref(&mail));
        assert!(matches!(&blocks[0], ContentBlock::Text(t) if t == "look"));
        assert!(
            matches!(&blocks[1], ContentBlock::Image(i) if matches!(&i.source, ImageSource::Bytes(b) if b == b"\x89PNG"))
        );
        assert!(matches!(&blocks[2], ContentBlock::Text(t) if t.contains("gone.png")));
        let content = result_content(&[mail]);
        assert!(matches!(&content[0], ToolResultContent::Image(_)));
        assert!(matches!(&content[1], ToolResultContent::Text(t) if t.contains("unavailable")));
    }
}

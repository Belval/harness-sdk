//! Images on their way to a model: attached to mail (the mailbox keeps their paths, and the bytes
//! are read only when the receiving agent takes the mail, as image blocks of its next model input)
//! or returned by a tool (`read`, MCP). Each is fitted to the provider's limits ([`fit`]) before it
//! enters the conversation, as one image over them fails every later call that resends it; one
//! already in history is replaced by a note once a call fails on it ([`drop_oversized`]).

use std::io::Cursor;
use std::path::Path;

use async_trait::async_trait;
use image::imageops::FilterType;
use image::{DynamicImage, ImageReader};
use strands_agents::hooks::AfterToolCallEvent;
use strands_agents::types::{ContentBlock, ImageBlock, ImageFormat, ImageSource, Message, ToolResultContent};
use strands_agents::{Agent, Plugin};

use crate::runtime::Mail;

/// Longest edge of an image sent to a model. Claude 4.7 and later see at most 2576 px on the long
/// edge (older models 1568 px) and downscale anything larger, and once a request holds more than
/// 20 images, any image over 3000 px is rejected ("many-image requests"), images from earlier
/// turns and inside tool results included; a tool-result image over the limits is rejected
/// rather than downscaled (<https://docs.claude.com/en/docs/build-with-claude/vision>). Bedrock's
/// Converse API refuses any image over 8000 px (`Message.content`).
pub const MAX_IMAGE_EDGE: u32 = 2576;

/// Largest image sent to a model: Bedrock's limit for one image, 3.75 MB (Converse
/// `Message.content`), which is Anthropic's 5 MB base64-encoded on Bedrock.
pub const MAX_IMAGE_BYTES: u64 = 3_750_000;

/// The note an image in history becomes once a model call failed on it.
pub const REMOVED_NOTE: &str = "[image removed: over the model's size limit]";

/// Edges below this are not worth sending: [`fit`] gives up instead.
const MIN_EDGE: u32 = 64;

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

/// Check that `path` can be attached: a supported format and readable. Its size is no limit, as
/// [`load`] fits it.
pub fn check(path: &Path) -> Result<(), String> {
    if format_of(path).is_none() {
        return Err(format!("{}: not a png, jpg, gif or webp image", path.display()));
    }
    std::fs::metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(())
}

/// The image block for the file at `path`, fitted to the model's limits.
pub fn load(path: &Path) -> Result<ImageBlock, String> {
    check(path)?;
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    fit(image_block(format_of(path).expect("checked"), bytes))
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Width and height of an encoded image, read from its header.
fn dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| error.to_string())?
        .into_dimensions()
        .map_err(|error| format!("not a readable image: {error}"))
}

/// `image` unchanged when it is within [`MAX_IMAGE_EDGE`] and [`MAX_IMAGE_BYTES`], else
/// downscaled and re-encoded until it is (as PNG, or JPEG when that is what fits or what it was).
/// An image that cannot be decoded is an error, as the model would refuse it too.
pub fn fit(image: ImageBlock) -> Result<ImageBlock, String> {
    let ImageSource::Bytes(bytes) = &image.source else {
        return Ok(image);
    };
    let (width, height) = dimensions(bytes)?;
    if width.max(height) <= MAX_IMAGE_EDGE && bytes.len() as u64 <= MAX_IMAGE_BYTES {
        return Ok(image);
    }
    let decoded = image::load_from_memory(bytes).map_err(|error| format!("not a readable image: {error}"))?;
    let photo = matches!(image.format, ImageFormat::Jpeg | ImageFormat::Jpg);
    let mut edge = width.max(height).min(MAX_IMAGE_EDGE);
    while edge >= MIN_EDGE {
        let scaled = match width.max(height) > edge {
            true => decoded.resize(edge, edge, FilterType::Triangle),
            false => decoded.clone(),
        };
        if !photo {
            let png = encode(&scaled, image::ImageFormat::Png)?;
            if png.len() as u64 <= MAX_IMAGE_BYTES {
                return Ok(image_block(ImageFormat::Png, png));
            }
        }
        let jpeg = encode(
            &DynamicImage::ImageRgb8(scaled.to_rgb8()),
            image::ImageFormat::Jpeg,
        )?;
        if jpeg.len() as u64 <= MAX_IMAGE_BYTES {
            return Ok(image_block(ImageFormat::Jpeg, jpeg));
        }
        edge = edge * 3 / 4;
    }
    Err(format!(
        "{width}x{height} px does not fit in {MAX_IMAGE_BYTES} bytes"
    ))
}

fn encode(image: &DynamicImage, format: image::ImageFormat) -> Result<Vec<u8>, String> {
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, format)
        .map_err(|error| error.to_string())?;
    Ok(bytes.into_inner())
}

fn image_block(format: ImageFormat, bytes: Vec<u8>) -> ImageBlock {
    ImageBlock {
        format,
        source: ImageSource::Bytes(bytes),
    }
}

/// Fits every image a tool returns ([`fit`]); one that cannot be becomes a note saying why.
pub struct FitImages;

#[async_trait]
impl Plugin for FitImages {
    fn name(&self) -> &str {
        "swarm:fit-images"
    }

    async fn init_agent(&self, agent: &Agent) -> strands_agents::Result<()> {
        agent.hooks().add_sync_callback(|event: &mut AfterToolCallEvent| {
            for item in &mut event.result.content {
                if let ToolResultContent::Image(image) = item {
                    *item = match fit(image.clone()) {
                        Ok(image) => ToolResultContent::Image(image),
                        Err(error) => ToolResultContent::Text(format!("[image removed: {error}]")),
                    };
                }
            }
            Ok(())
        });
        Ok(())
    }
}

/// Whether a model error names an image limit, like Anthropic's `messages.1.content.51.image.
/// source.base64.data: At least one of the image dimensions exceed max allowed size for
/// many-image requests: 3000 pixels` or `image exceeds 5 MB maximum`.
pub fn names_image_limit(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("image")
        && ["pixel", "exceed", "too large", "maximum"]
            .iter()
            .any(|word| error.contains(word))
}

/// Replace the images in `messages` that a model call failed on with `error` (see
/// [`names_image_limit`]) by [`REMOVED_NOTE`], so the next call can succeed; returns how many.
/// Those are the images over the pixel limit the error states (`… 3000 pixels`), else over
/// [`fit`]'s limits, and all of them when none is: the error names a limit, so keeping every
/// image would fail every later call.
pub fn drop_oversized(messages: &mut [Message], error: &str) -> usize {
    let stated = regex::Regex::new(r"(\d+)\s*pixels")
        .expect("valid regex")
        .captures(error)
        .and_then(|captures| captures[1].parse::<u32>().ok());
    let edge = stated.unwrap_or(MAX_IMAGE_EDGE);
    let over = |image: &ImageBlock| match &image.source {
        ImageSource::Bytes(bytes) => {
            bytes.len() as u64 > MAX_IMAGE_BYTES
                || dimensions(bytes).map_or(true, |(width, height)| width.max(height) > edge)
        }
        _ => false,
    };
    match replace_images(messages, over) {
        0 => replace_images(messages, |_| true),
        removed => removed,
    }
}

/// Replace each image in `messages` for which `remove` holds by [`REMOVED_NOTE`]; returns how many.
fn replace_images(messages: &mut [Message], remove: impl Fn(&ImageBlock) -> bool) -> usize {
    let mut removed = 0;
    for block in messages.iter_mut().flat_map(|message| message.content.iter_mut()) {
        match block {
            ContentBlock::Image(image) if remove(image) => {
                *block = ContentBlock::Text(REMOVED_NOTE.to_owned());
                removed += 1;
            }
            ContentBlock::ToolResult(result) => {
                for item in &mut result.content {
                    if matches!(item, ToolResultContent::Image(image) if remove(image)) {
                        *item = ToolResultContent::Text(REMOVED_NOTE.to_owned());
                        removed += 1;
                    }
                }
            }
            _ => {}
        }
    }
    removed
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
    use strands_agents::types::{Role, ToolResultBlock};

    /// A real PNG of noise, which compresses badly.
    fn noise_png(width: u32, height: u32) -> Vec<u8> {
        let mut seed = 0x2545_f491_u32;
        let image = image::RgbImage::from_fn(width, height, |_, _| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let [r, g, b, _] = seed.to_le_bytes();
            image::Rgb([r, g, b])
        });
        encode(&DynamicImage::ImageRgb8(image), image::ImageFormat::Png).unwrap()
    }

    fn bytes_of(image: &ImageBlock) -> &[u8] {
        match &image.source {
            ImageSource::Bytes(bytes) => bytes,
            other => panic!("not bytes: {other:?}"),
        }
    }

    #[test]
    fn knows_image_formats_by_extension() {
        assert_eq!(format_of(Path::new("a/b.PNG")), Some(ImageFormat::Png));
        assert_eq!(format_of(Path::new("x.jpeg")), Some(ImageFormat::Jpeg));
        assert_eq!(format_of(Path::new("x.txt")), None);
        assert_eq!(format_of(Path::new("png")), None);
    }

    #[test]
    fn checks_format_and_existence() {
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
    }

    #[test]
    fn fit_keeps_an_image_within_the_limits_as_it_is() {
        let png = crate::tui::images::png(40, 30);
        let image = image_block(ImageFormat::Png, png.clone());
        assert_eq!(fit(image.clone()).unwrap(), image);
        assert!(fit(image_block(ImageFormat::Png, b"\x89PNG".to_vec()))
            .unwrap_err()
            .contains("not a readable image"));
    }

    #[test]
    fn fit_downscales_an_oversized_png_within_pixels_and_bytes() {
        // 2700 px wide and over 3.75 MB as noise: too large on both counts.
        let png = noise_png(2700, 500);
        assert!(png.len() as u64 > MAX_IMAGE_BYTES);
        let fitted = fit(image_block(ImageFormat::Png, png)).unwrap();
        let bytes = bytes_of(&fitted);
        assert!(bytes.len() as u64 <= MAX_IMAGE_BYTES, "{} bytes", bytes.len());
        let (width, height) = dimensions(bytes).unwrap();
        assert!(width <= MAX_IMAGE_EDGE && height <= MAX_IMAGE_EDGE);
        assert_eq!(width, MAX_IMAGE_EDGE, "only as small as needed");
        assert_eq!(height, 500 * MAX_IMAGE_EDGE / 2700, "aspect ratio kept");
    }

    #[test]
    fn a_tool_returning_an_oversized_image_gets_it_fitted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.png");
        std::fs::write(&path, crate::tui::images::png(4000, 100)).unwrap();
        let fitted = load(&path).unwrap();
        assert_eq!(fitted.format, ImageFormat::Png);
        assert_eq!(dimensions(bytes_of(&fitted)).unwrap(), (MAX_IMAGE_EDGE, 64));
    }

    #[test]
    fn recognises_image_limit_errors() {
        assert!(names_image_limit(
            "The model returned the following errors: messages.1.content.51.image.source.base64.data: \
             At least one of the image dimensions exceed max allowed size for many-image requests: 3000 pixels"
        ));
        assert!(names_image_limit(
            "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum"
        ));
        assert!(!names_image_limit("Input is too long for requested model."));
        assert!(!names_image_limit("ThrottlingException: Too many requests"));
    }

    #[test]
    fn drop_oversized_repairs_a_history_whose_call_failed_on_an_image() {
        let small = image_block(ImageFormat::Png, crate::tui::images::png(100, 100));
        let large = image_block(ImageFormat::Png, crate::tui::images::png(3100, 20));
        let mut result = ToolResultBlock::success("t1", vec![]);
        result.content = vec![
            ToolResultContent::Text("read".into()),
            ToolResultContent::Image(large.clone()),
        ];
        let mut messages = vec![
            Message::new(
                Role::User,
                vec![
                    ContentBlock::Text("look".into()),
                    ContentBlock::Image(small.clone()),
                ],
            ),
            Message::new(Role::User, vec![ContentBlock::ToolResult(result)]),
        ];
        let error = "messages.1.content.0.image.source.base64.data: At least one of the image \
                     dimensions exceed max allowed size for many-image requests: 3000 pixels";
        assert_eq!(drop_oversized(&mut messages, error), 1);
        assert!(matches!(&messages[0].content[1], ContentBlock::Image(image) if *image == small));
        let ContentBlock::ToolResult(result) = &messages[1].content[0] else {
            panic!("not a tool result");
        };
        assert!(matches!(&result.content[1], ToolResultContent::Text(note) if note == REMOVED_NOTE));
        // Nothing over the limit is left; an error naming one anyway drops every image.
        assert_eq!(drop_oversized(&mut messages, "image exceeds 5 MB maximum"), 1);
        assert!(matches!(&messages[0].content[1], ContentBlock::Text(note) if note == REMOVED_NOTE));
    }

    #[test]
    fn attachments_become_image_blocks_or_notes() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("ok.png");
        let png = crate::tui::images::png(8, 6);
        std::fs::write(&ok, &png).unwrap();
        let mail = Mail {
            from: "user".into(),
            from_label: "user".into(),
            kind: "message",
            text: "look".into(),
            images: vec![ok, dir.path().join("gone.png")],
        };
        let blocks = input_blocks("look".into(), std::slice::from_ref(&mail));
        assert!(matches!(&blocks[0], ContentBlock::Text(t) if t == "look"));
        assert!(matches!(&blocks[1], ContentBlock::Image(i) if bytes_of(i) == png));
        assert!(matches!(&blocks[2], ContentBlock::Text(t) if t.contains("gone.png")));
        let content = result_content(&[mail]);
        assert!(matches!(&content[0], ToolResultContent::Image(_)));
        assert!(matches!(&content[1], ToolResultContent::Text(t) if t.contains("unavailable")));
    }
}

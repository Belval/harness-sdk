//! Inline images in the log: decoded thumbnails for the terminal's graphics protocol (kitty,
//! iTerm2, sixel) or unicode halfblocks. Only the newest few stay decoded; older ones, and any
//! without a picker (tests, the reload self-test), show just their `[image: …]` placeholder.

use std::collections::VecDeque;
use std::fmt;

use ratatui::layout::Size;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::Protocol;
use ratatui_image::Resize;

/// Most rows a thumbnail takes in the log.
pub const MAX_ROWS: u16 = 12;
/// Most columns a thumbnail takes in the log.
pub const MAX_COLS: u16 = 60;
/// Thumbnails kept decoded; older images fall back to their placeholder.
pub const KEEP_DECODED: usize = 4;

/// The decoded thumbnails, and the picker that encodes them for this terminal.
#[derive(Default)]
pub struct Images {
    picker: Option<Picker>,
    decoded: VecDeque<(u64, Protocol)>,
    next: u64,
}

impl fmt::Debug for Images {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Images")
            .field("picker", &self.picker.as_ref().map(Picker::protocol_type))
            .field(
                "decoded",
                &self.decoded.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// What decoding one image found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoded {
    /// Key for [`Images::get`].
    pub id: u64,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
}

impl Images {
    /// Thumbnails encoded with `picker`.
    pub fn with_picker(picker: Picker) -> Self {
        Self {
            picker: Some(picker),
            ..Self::default()
        }
    }

    /// The picker for the terminal on stdout: halfblocks inside tmux (whose passthrough of
    /// graphics protocols is unreliable) or when the terminal does not answer the query.
    pub fn detect() -> Self {
        let in_tmux = std::env::var_os("TMUX").is_some_and(|v| !v.is_empty());
        let picker = match in_tmux {
            true => Picker::halfblocks(),
            false => Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks()),
        };
        Self::with_picker(picker)
    }

    /// Decode `bytes` (png, jpeg, gif or webp) and, with a picker, keep its thumbnail.
    pub fn add(&mut self, bytes: &[u8]) -> Result<Decoded, String> {
        let image = image::load_from_memory(bytes).map_err(|error| error.to_string())?;
        let id = self.next;
        self.next += 1;
        let decoded = Decoded {
            id,
            width: image.width(),
            height: image.height(),
        };
        if let Some(picker) = &self.picker {
            let size = Size::new(MAX_COLS, MAX_ROWS);
            if let Ok(protocol) = picker.new_protocol(image, size, Resize::Fit(None)) {
                self.decoded.push_back((id, protocol));
                while self.decoded.len() > KEEP_DECODED {
                    self.decoded.pop_front();
                }
            }
        }
        Ok(decoded)
    }

    /// The thumbnail of image `id`, while it is kept.
    pub fn get(&self, id: u64) -> Option<&Protocol> {
        self.decoded.iter().find(|(key, _)| *key == id).map(|(_, p)| p)
    }

    /// Rows the thumbnail of image `id` takes below its placeholder (0 once dropped).
    pub fn rows(&self, id: u64) -> u16 {
        self.get(id).map_or(0, |protocol| protocol.size().height)
    }
}

#[cfg(test)]
pub(crate) fn png(width: u32, height: u32) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x * 8) as u8, (y * 8) as u8, 128])
    });
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    bytes.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_sizes_without_a_picker_and_keeps_no_thumbnail() {
        let mut images = Images::default();
        let decoded = images.add(&png(30, 20)).unwrap();
        assert_eq!((decoded.width, decoded.height), (30, 20));
        assert!(images.get(decoded.id).is_none());
        assert_eq!(images.rows(decoded.id), 0);
        assert!(images.add(b"not an image").is_err());
    }

    #[test]
    fn keeps_bounded_thumbnails_of_the_newest_images() {
        let mut images = Images::with_picker(Picker::halfblocks());
        let ids: Vec<u64> = (0..6).map(|_| images.add(&png(400, 400)).unwrap().id).collect();
        assert!(images.get(ids[0]).is_none() && images.get(ids[1]).is_none());
        for id in &ids[2..] {
            let rows = images.rows(*id);
            assert!(rows > 0 && rows <= MAX_ROWS, "{rows}");
            assert!(images.get(*id).unwrap().size().width <= MAX_COLS);
        }
    }
}

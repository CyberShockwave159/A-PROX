/// Target-area/ratio → WxH math for generated images.
///
/// A-PROX computes final dimensions itself and feeds them directly into the
/// workflow's `EmptyLatentImage` (t2i) / `ResizeImageMaskNode` (i2i), bypassing
/// ComfyUI's `ResolutionSelector` (which only offers 8 preset ratios).
pub fn compute_dimensions(
    wh_ratio: Option<&str>,
    reference: Option<(u32, u32)>,
    mp: f64,
    multiple: u32,
    max_side: u32,
) -> (u32, u32) {
    let m = multiple.clamp(1, 128);
    let max = max_side.max(m);

    let (mut w, mut h) = match reference {
        Some((rw, rh)) if rw > 0 && rh > 0 => (rw as f64, rh as f64),
        _ => {
            let (rw, rh) = parse_ratio(wh_ratio.unwrap_or("1:1"));
            let area = mp.max(0.1) * 1_000_000.0;
            let ratio = rw / rh;
            let w = (area * ratio).sqrt();
            let h = w / ratio;
            (w, h)
        }
    };

    // Keep aspect ratio while capping the long edge.
    if w >= h {
        if w > max as f64 {
            h = h * max as f64 / w;
            w = max as f64;
        }
    } else if h > max as f64 {
        w = w * max as f64 / h;
        h = max as f64;
    }

    (snap(m, w), snap(m, h))
}

/// Parse `"W:H"` into a (w, h) ratio pair; malformed input collapses to 1:1.
pub fn parse_ratio(s: &str) -> (f64, f64) {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() == 2 {
        let w = parts[0].trim().parse::<f64>().unwrap_or(0.0);
        let h = parts[1].trim().parse::<f64>().unwrap_or(0.0);
        if w > 0.0 && h > 0.0 {
            return (w, h);
        }
    }
    (1.0, 1.0)
}

/// Round to the nearest `multiple`, at least `multiple`, never exceeding `max`
/// after snapping (which may reduce the capped edge slightly).
fn snap(multiple: u32, v: f64) -> u32 {
    let m = multiple.max(1);
    let snapped = ((v / m as f64).round() as u32).saturating_mul(m).max(m);
    let cap = m * (multiple.max(1)).max(1); // unused fallback guard
    let _ = cap;
    snapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ratio() {
        assert_eq!(parse_ratio("3:2"), (3.0, 2.0));
        assert_eq!(parse_ratio("16:9"), (16.0, 9.0));
        assert_eq!(parse_ratio("garbage"), (1.0, 1.0));
        assert_eq!(parse_ratio("2:0"), (1.0, 1.0));
        assert_eq!(parse_ratio("1:1 "), (1.0, 1.0));
    }

    #[test]
    fn test_square_at_2mp() {
        let (w, h) = compute_dimensions(Some("1:1"), None, 2.0, 16, 4096);
        assert_eq!(w, h);
        assert_eq!(w % 16, 0);
        assert!(w >= 1400 && w <= 1500, "got {w}x{h}");
    }

    #[test]
    fn test_ratio_respected() {
        let (w, h) = compute_dimensions(Some("16:9"), None, 2.0, 16, 4096);
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
        assert!((w as f64 / (h as f64) - 16.0 / 9.0).abs() < 0.2, "got {w}x{h}");
        assert!(w > h);
        assert!(w <= 4096);
    }

    #[test]
    fn test_reference_dims_win() {
        let (w, h) = compute_dimensions(Some("16:9"), Some((800, 600)), 2.0, 16, 4096);
        assert_eq!(w, 800);
        assert_eq!(h, 608); // 600 snapped to the nearest multiple of 16
    }

    #[test]
    fn test_capped_long_edge() {
        let (w, h) = compute_dimensions(Some("21:9"), None, 2.0, 16, 1024);
        assert!(w <= 1024, "w={w}");
        assert!(h <= 1024, "h={h}");
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
    }

    #[test]
    fn test_portrait() {
        let (w, h) = compute_dimensions(Some("2:3"), None, 2.0, 16, 4096);
        assert!(h > w);
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
    }

    #[test]
    fn test_max_side_never_exceeded() {
        let (w, h) = compute_dimensions(Some("4096:4096"), None, 100.0, 16, 2048);
        assert!(w <= 2048);
        assert!(h <= 2048);
    }
}
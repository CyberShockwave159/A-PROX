use scraper::{Html, Selector};

pub struct CleanPage {
    pub title: String,
    pub content: String,
}

/// Strips boilerplates (nav, footer, script, style, ads) and extracts readable markdown/text
pub fn clean_html(html_str: &str) -> CleanPage {
    let document = Html::parse_document(html_str);

    // Extract title
    let title_selector = Selector::parse("title").unwrap();
    let title = document
        .select(&title_selector)
        .next()
        .map(|el| el.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
        .trim()
        .to_string();

    // Elements to extract text from
    let content_selector = Selector::parse(
        "article, main, .content, .post-content, #content, body"
    ).unwrap();

    let root_el = document.select(&content_selector).next();

    let text = if let Some(root) = root_el {
        extract_clean_text(&root)
    } else {
        extract_clean_text(&document.root_element())
    };

    CleanPage {
        title,
        content: text,
    }
}

fn extract_clean_text(element: &scraper::ElementRef) -> String {
    let blacklist = [
        "script", "style", "nav", "footer", "header", "aside",
        "iframe", "noscript", "svg", "form", "button",
    ];

    let mut out = String::new();
    let p_selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, pre, code, blockquote").unwrap();

    for el in element.select(&p_selector) {
        // Check if element or any ancestor is in blacklist
        let tag = el.value().name();
        if blacklist.contains(&tag) {
            continue;
        }

        let mut is_blacklisted = false;
        let mut parent = el.parent();
        while let Some(p) = parent {
            if let Some(p_elem) = p.value().as_element() {
                if blacklist.contains(&p_elem.name()) {
                    is_blacklisted = true;
                    break;
                }
            }
            parent = p.parent();
        }

        if is_blacklisted {
            continue;
        }

        let text = el.text().collect::<Vec<_>>().join(" ").trim().to_string();
        if text.len() > 15 {
            if tag.starts_with('h') {
                out.push_str("\n### ");
                out.push_str(&text);
                out.push_str("\n\n");
            } else if tag == "li" {
                out.push_str("- ");
                out.push_str(&text);
                out.push('\n');
            } else {
                out.push_str(&text);
                out.push_str("\n\n");
            }
        }
    }

    out.trim().to_string()
}

import argparse
import hashlib
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


BASE_URL = "https://cellphones.com.vn"
CATEGORY_URL = "https://cellphones.com.vn/mobile.html"
SITEMAP_INDEX_URL = "https://cellphones.com.vn/sitemap/sitemap_index.xml?v=google"

OUTPUT_DIR = Path("data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
}

PHONE_KEYWORDS = [
    "dien-thoai",
    "iphone",
    "samsung-galaxy",
    "xiaomi",
    "redmi",
    "poco",
    "oppo",
    "realme",
    "vivo",
    "honor",
    "nubia",
    "tecno",
    "infinix",
    "nokia",
    "nothing-phone",
    "oneplus",
    "motorola",
]

EXCLUDE_KEYWORDS = [
    "op-lung",
    "bao-da",
    "cuong-luc",
    "kinh-cuong-luc",
    "sac",
    "cap-sac",
    "tai-nghe",
    "loa",
    "dong-ho",
    "watch",
    "ipad",
    "tablet",
    "laptop",
    "macbook",
    "may-tinh",
    "phu-kien",
    "sim",
    "esim",
    "camera",
    "may-anh",
    "tivi",
    "man-hinh",
]

BRANDS = [
    "Apple",
    "iPhone",
    "Samsung",
    "Xiaomi",
    "Redmi",
    "POCO",
    "OPPO",
    "Realme",
    "Vivo",
    "HONOR",
    "Nubia",
    "Tecno",
    "Infinix",
    "Nokia",
    "Nothing",
    "OnePlus",
    "Motorola",
]

COLOR_WORDS = [
    "đen",
    "trắng",
    "xanh",
    "đỏ",
    "vàng",
    "tím",
    "hồng",
    "bạc",
    "xám",
    "cam",
    "nâu",
    "graphite",
    "titan",
    "blue",
    "black",
    "white",
    "silver",
    "gold",
    "pink",
    "purple",
    "green",
]

SPEC_LABELS = [
    "Kích thước màn hình",
    "Công nghệ màn hình",
    "Độ phân giải màn hình",
    "Tính năng màn hình",
    "Tần số quét",
    "Độ sáng tối đa",
    "Camera sau",
    "Camera trước",
    "Chipset",
    "Chip xử lý",
    "CPU",
    "GPU",
    "Công nghệ NFC",
    "Dung lượng RAM",
    "RAM",
    "Bộ nhớ trong",
    "Dung lượng lưu trữ",
    "Pin",
    "Dung lượng pin",
    "Sạc nhanh",
    "Thẻ SIM",
    "Hệ điều hành",
    "Kích thước",
    "Trọng lượng",
    "Chất liệu",
    "Cổng sạc",
    "Bluetooth",
    "Wi-Fi",
    "Kháng nước",
    "Bảo mật",
]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def sleep_delay(delay: float):
    time.sleep(delay + random.uniform(0.3, 1.2))


def fetch(session: requests.Session, url: str, delay: float = 1.5) -> Optional[str]:
    try:
        sleep_delay(delay)
        response = session.get(url, timeout=25)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text" not in content_type and "xml" not in content_type and "html" not in content_type:
            return None

        return response.text

    except Exception as e:
        print(f"[FETCH ERROR] {url} -> {e}")
        return None


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()


def clean_multiline(text: str) -> str:
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        line = clean_text(line)
        if line:
            lines.append(line)

    return "\n".join(lines)


def absolute_url(href: str) -> str:
    return urljoin(BASE_URL, href)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def slug_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.split("/")[-1]
    return name.replace(".html", "")


def is_phone_product_url(url: str) -> bool:
    url = normalize_url(url).lower()

    if not url.startswith(BASE_URL):
        return False

    if not url.endswith(".html"):
        return False

    slug = slug_from_url(url)

    if any(k in slug for k in EXCLUDE_KEYWORDS):
        return False

    return any(k in slug for k in PHONE_KEYWORDS)


def parse_price_to_int(price_text: str) -> Optional[int]:
    if not price_text:
        return None

    digits = re.sub(r"[^\d]", "", price_text)
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def extract_prices(text: str) -> List[str]:
    patterns = [
        r"\d{1,3}(?:\.\d{3}){2,}\s*[đ₫]",
        r"\d{1,3}(?:,\d{3}){2,}\s*[đ₫]",
    ]

    prices = []

    for pattern in patterns:
        prices.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    cleaned = []
    seen = set()

    for price in prices:
        price = clean_text(price)
        if price not in seen:
            seen.add(price)
            cleaned.append(price)

    return cleaned


def infer_brand(name: str) -> str:
    lower = name.lower()

    for brand in BRANDS:
        if brand.lower() in lower:
            if brand.lower() == "iphone":
                return "Apple"
            return brand

    return ""


def extract_ram_storage(text: str) -> Dict[str, str]:
    text = clean_text(text)

    ram = ""
    storage = ""

    ram_match = re.search(r"(?i)(\d+\s*GB)\s*RAM", text)
    if ram_match:
        ram = ram_match.group(1).replace(" ", "")

    storage_matches = re.findall(r"(?i)(\d+\s*(?:GB|TB))", text)
    normalized = [x.replace(" ", "").upper() for x in storage_matches]

    if ram:
        normalized_without_ram = [x for x in normalized if x != ram.upper()]
    else:
        normalized_without_ram = normalized

    if normalized_without_ram:
        storage = normalized_without_ram[-1]

    if not ram:
        # Tên CellphoneS thường là: "S26 Ultra 12GB 256GB"
        if len(normalized) >= 2:
            ram = normalized[0]
            storage = normalized[1]
        elif len(normalized) == 1:
            storage = normalized[0]

    return {
        "ram": ram,
        "storage": storage,
    }


def redact_personal_info(text: str) -> str:
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_ẨN]", text)
    text = re.sub(r"(?<!\d)(0|\+84)[\d\s\.\-]{8,13}(?!\d)", "[SĐT_ẨN]", text)
    return text


def parse_sitemap_urls(xml_text: str) -> List[str]:
    urls = []

    try:
        root = ET.fromstring(xml_text)
        for loc in root.findall(".//{*}loc"):
            if loc.text:
                urls.append(clean_text(loc.text))
    except Exception as e:
        print(f"[SITEMAP PARSE ERROR] {e}")

    return urls


def discover_product_urls_from_sitemap(
    session: requests.Session,
    max_products: int,
    delay: float,
) -> List[str]:
    product_urls = []
    seen = set()

    index_xml = fetch(session, SITEMAP_INDEX_URL, delay=delay)

    if not index_xml:
        return []

    sitemap_urls = parse_sitemap_urls(index_xml)

    product_sitemaps = [
        u for u in sitemap_urls
        if "product-sitemap" in u
    ]

    print(f"[INFO] Found {len(product_sitemaps)} product sitemaps")

    for sitemap_url in product_sitemaps:
        if len(product_urls) >= max_products:
            break

        xml_text = fetch(session, sitemap_url, delay=delay)
        if not xml_text:
            continue

        urls = parse_sitemap_urls(xml_text)

        for url in urls:
            url = normalize_url(url)

            if url in seen:
                continue

            if not is_phone_product_url(url):
                continue

            seen.add(url)
            product_urls.append(url)

            if len(product_urls) >= max_products:
                break

    return product_urls


def discover_product_urls_from_category(
    session: requests.Session,
    max_products: int,
    delay: float,
) -> List[str]:
    html = fetch(session, CATEGORY_URL, delay=delay)

    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    urls = []
    seen = set()

    for a in soup.find_all("a", href=True):
        url = normalize_url(absolute_url(a["href"]))

        if url in seen:
            continue

        if is_phone_product_url(url):
            seen.add(url)
            urls.append(url)

        if len(urls) >= max_products:
            break

    return urls


def extract_json_ld(soup: BeautifulSoup) -> List[dict]:
    results = []

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        raw = raw.strip()

        if not raw:
            continue

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                results.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue

    return results


def find_product_jsonld(json_ld_items: List[dict]) -> Optional[dict]:
    for item in json_ld_items:
        item_type = item.get("@type", "")

        if isinstance(item_type, list):
            item_type = " ".join(item_type)

        if "Product" in str(item_type):
            return item

        graph = item.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                node_type = node.get("@type", "")
                if isinstance(node_type, list):
                    node_type = " ".join(node_type)

                if "Product" in str(node_type):
                    return node

    return None


def extract_name(soup: BeautifulSoup, product_jsonld: Optional[dict]) -> str:
    if product_jsonld and product_jsonld.get("name"):
        return clean_text(product_jsonld["name"])

    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))

    meta = soup.find("meta", property="og:title")
    if meta and meta.get("content"):
        return clean_text(meta["content"])

    return ""


def extract_images(soup: BeautifulSoup, product_jsonld: Optional[dict]) -> List[str]:
    images = []

    if product_jsonld:
        img = product_jsonld.get("image")
        if isinstance(img, str):
            images.append(img)
        elif isinstance(img, list):
            images.extend([x for x in img if isinstance(x, str)])

    for meta_prop in ["og:image", "twitter:image"]:
        meta = soup.find("meta", property=meta_prop) or soup.find("meta", attrs={"name": meta_prop})
        if meta and meta.get("content"):
            images.append(meta["content"])

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        alt = img.get("alt", "")

        if not src:
            continue

        src = absolute_url(src)

        if alt and "cellphones" not in src.lower():
            images.append(src)

    cleaned = []
    seen = set()

    for img in images:
        img = clean_text(img)
        if img and img not in seen:
            seen.add(img)
            cleaned.append(img)

    return cleaned[:20]


def extract_rating(soup: BeautifulSoup, product_jsonld: Optional[dict]) -> Dict:
    rating_value = None
    review_count = None

    if product_jsonld:
        aggregate = product_jsonld.get("aggregateRating") or {}

        if isinstance(aggregate, dict):
            rating_value = aggregate.get("ratingValue") or aggregate.get("rating")
            review_count = aggregate.get("reviewCount") or aggregate.get("ratingCount")

    page_text = soup.get_text("\n", strip=True)

    match = re.search(
        r"(?P<rating>\d+(?:[.,]\d+)?)\s*\(\s*(?P<count>\d+)\s*đánh giá\s*\)",
        page_text,
        flags=re.IGNORECASE,
    )

    if match:
        rating_value = match.group("rating").replace(",", ".")
        review_count = match.group("count")

    return {
        "rating_value": str(rating_value) if rating_value is not None else "",
        "review_count": int(review_count) if str(review_count or "").isdigit() else None,
    }


def extract_price(soup: BeautifulSoup, product_jsonld: Optional[dict]) -> Dict:
    current_price = ""
    old_price = ""
    discount = ""

    if product_jsonld:
        offers = product_jsonld.get("offers")

        if isinstance(offers, dict):
            price = offers.get("price")
            currency = offers.get("priceCurrency", "VND")

            if price:
                current_price = f"{price} {currency}"

        elif isinstance(offers, list) and offers:
            first_offer = offers[0]
            if isinstance(first_offer, dict) and first_offer.get("price"):
                current_price = f"{first_offer.get('price')} {first_offer.get('priceCurrency', 'VND')}"

    page_text = soup.get_text(" ", strip=True)
    prices = extract_prices(page_text)

    if not current_price and prices:
        current_price = prices[0]

    if len(prices) >= 2:
        p0 = parse_price_to_int(prices[0])
        for p in prices[1:]:
            pi = parse_price_to_int(p)
            if p0 and pi and pi > p0:
                old_price = p
                break

    discount_match = re.search(r"Giảm\s+\d+%", page_text, flags=re.IGNORECASE)
    if discount_match:
        discount = discount_match.group(0)

    return {
        "current_price": current_price,
        "old_price": old_price,
        "discount": discount,
    }


def extract_highlights(soup: BeautifulSoup) -> List[str]:
    text = soup.get_text("\n", strip=True)
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]

    highlights = []
    capture = False

    for line in lines:
        lower = line.lower()

        if "tính năng nổi bật" in lower:
            capture = True
            continue

        if capture and (
            "thông số kỹ thuật" in lower
            or "phiên bản" in lower
            or "màu sắc" in lower
            or "đặc điểm nổi bật" in lower
        ):
            break

        if capture:
            if len(line) >= 25 and line not in highlights:
                highlights.append(line)

    return highlights[:8]


def extract_specs(soup: BeautifulSoup) -> Dict[str, str]:
    specs = {}

    # Cách 1: nếu trang có table thông số
    for row in soup.select("table tr"):
        cols = row.find_all(["td", "th"])
        if len(cols) >= 2:
            key = clean_text(cols[0].get_text(" ", strip=True))
            value = clean_text(cols[1].get_text(" ", strip=True))

            if key and value and len(key) <= 80:
                specs[key] = value

    if specs:
        return specs

    # Cách 2: parse từ text block "Thông số kỹ thuật"
    text = soup.get_text("\n", strip=True)
    text = clean_multiline(text)

    start = text.lower().find("thông số kỹ thuật")
    if start == -1:
        return specs

    segment = text[start:]

    stop_keywords = [
        "phiên bản",
        "màu sắc",
        "trả góp",
        "đặc điểm nổi bật",
        "đánh giá",
        "mua ngay",
    ]

    stop_positions = []
    lower_segment = segment.lower()

    for kw in stop_keywords:
        pos = lower_segment.find(kw)
        if pos > 20:
            stop_positions.append(pos)

    if stop_positions:
        segment = segment[:min(stop_positions)]

    positions = []

    for label in SPEC_LABELS:
        match = re.search(re.escape(label), segment, flags=re.IGNORECASE)
        if match:
            positions.append((match.start(), match.end(), label))

    positions.sort(key=lambda x: x[0])

    for idx, (_, end, label) in enumerate(positions):
        next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(segment)
        value = segment[end:next_start]
        value = clean_multiline(value)
        value = re.sub(r"^[:\-\s]+", "", value).strip()

        if value:
            specs[label] = value

    return specs


def looks_like_variant(text: str) -> bool:
    text = clean_text(text)
    lower = text.lower()

    has_storage = bool(re.search(r"\d+\s*(gb|tb)", lower))
    has_phone_word = any(x in lower for x in ["iphone", "galaxy", "redmi", "poco", "oppo", "vivo", "honor", "realme", "nubia", "tecno"])

    return has_storage and has_phone_word and len(text) <= 120


def extract_variants(soup: BeautifulSoup, current_name: str) -> List[Dict]:
    variants = []
    seen = set()

    for a in soup.find_all("a", href=True):
        text = clean_text(a.get_text(" ", strip=True))
        href = normalize_url(absolute_url(a["href"]))

        if not text:
            continue

        if not looks_like_variant(text):
            continue

        if not is_phone_product_url(href):
            continue

        key = f"{text}|{href}"
        if key in seen:
            continue

        seen.add(key)

        prices = extract_prices(text)
        rs = extract_ram_storage(text)

        variants.append({
            "name": text,
            "url": href,
            "ram": rs["ram"],
            "storage": rs["storage"],
            "price": prices[0] if prices else "",
        })

    # Đảm bảo sản phẩm hiện tại cũng có trong variants
    current_rs = extract_ram_storage(current_name)
    current_url = ""

    if current_name and not any(v["name"].lower() == current_name.lower() for v in variants):
        variants.insert(0, {
            "name": current_name,
            "url": current_url,
            "ram": current_rs["ram"],
            "storage": current_rs["storage"],
            "price": "",
        })

    return variants[:20]


def extract_colors(soup: BeautifulSoup) -> List[Dict]:
    colors = []
    seen = set()

    color_regex = re.compile("|".join([re.escape(c) for c in COLOR_WORDS]), re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        text = clean_text(a.get_text(" ", strip=True))

        if not text:
            continue

        if not color_regex.search(text):
            continue

        prices = extract_prices(text)

        if len(text) > 100:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)

        color_name = text
        price = ""

        if prices:
            price = prices[0]
            color_name = clean_text(text.replace(price, ""))

        colors.append({
            "name": color_name,
            "price": price,
            "url": normalize_url(absolute_url(a["href"])),
        })

    return colors[:12]


def extract_description(soup: BeautifulSoup) -> str:
    # Ưu tiên các khối thường chứa bài viết mô tả
    selector_candidates = [
        ".box-product-information",
        ".cps-block-content",
        ".product-description",
        "[class*='product-information']",
        "[class*='description']",
    ]

    for selector in selector_candidates:
        block = soup.select_one(selector)
        if block:
            text = clean_multiline(block.get_text("\n", strip=True))
            if len(text) > 200:
                return text

    # Fallback theo heading "Đặc điểm nổi bật"
    full_text = soup.get_text("\n", strip=True)
    full_text = clean_multiline(full_text)

    lower = full_text.lower()
    start_candidates = [
        lower.find("đặc điểm nổi bật"),
        lower.find("đánh giá"),
    ]

    start_candidates = [x for x in start_candidates if x != -1]

    if not start_candidates:
        return ""

    start = min(start_candidates)

    stop_keywords = [
        "video đánh giá",
        "hỏi và đáp",
        "thông báo",
        "sản phẩm bạn quan tâm",
        "lên đầu",
        "liên hệ",
    ]

    end = len(full_text)
    lower_after = lower[start:]

    for kw in stop_keywords:
        pos = lower_after.find(kw)
        if pos > 300:
            end = min(end, start + pos)

    return full_text[start:end].strip()


def extract_comments_and_reviews(
    soup: BeautifulSoup,
    product_jsonld: Optional[dict],
    include_comments: bool,
) -> Dict:
    reviews = []
    comments = []

    if product_jsonld:
        raw_reviews = product_jsonld.get("review", [])

        if isinstance(raw_reviews, dict):
            raw_reviews = [raw_reviews]

        if isinstance(raw_reviews, list):
            for r in raw_reviews:
                if not isinstance(r, dict):
                    continue

                author = r.get("author", "")
                if isinstance(author, dict):
                    author = author.get("name", "")

                review_body = r.get("reviewBody") or r.get("description") or ""
                rating = ""

                review_rating = r.get("reviewRating")
                if isinstance(review_rating, dict):
                    rating = review_rating.get("ratingValue", "")

                item = {
                    "author": redact_personal_info(clean_text(str(author))),
                    "rating": str(rating),
                    "content": redact_personal_info(clean_text(review_body)),
                }

                if item["content"]:
                    reviews.append(item)

    if not include_comments:
        return {
            "reviews": reviews,
            "comments": comments,
            "comment_note": "Không thu thập nội dung bình luận vì include_comments=False.",
        }

    # Best effort: chỉ lấy comment/review/Q&A nếu server trả sẵn trong HTML
    selectors = [
        "[class*='comment']",
        "[class*='review']",
        "[class*='question']",
        "[class*='answer']",
        "[class*='rating']",
    ]

    seen = set()

    for selector in selectors:
        for block in soup.select(selector):
            text = clean_text(block.get_text(" ", strip=True))

            if not text:
                continue

            lower = text.lower()

            boilerplates = [
                "gửi câu hỏi",
                "hãy đặt câu hỏi",
                "cellphones sẽ phản hồi",
                "thông tin có thể thay đổi",
                "đăng nhập",
                "đăng ký",
                "yêu thích",
            ]

            if any(b in lower for b in boilerplates):
                continue

            if len(text) < 25 or len(text) > 800:
                continue

            text = redact_personal_info(text)

            if text in seen:
                continue

            seen.add(text)

            comments.append({
                "content": text,
            })

    note = ""
    if not comments:
        note = (
            "Không tìm thấy bình luận chi tiết trong HTML tĩnh. "
            "Có thể phần này được load bằng JavaScript/API riêng."
        )

    return {
        "reviews": reviews,
        "comments": comments[:30],
        "comment_note": note,
    }


def make_product_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def parse_product_page(
    html: str,
    url: str,
    include_comments: bool,
) -> Dict:
    soup = BeautifulSoup(html, "lxml")

    json_ld_items = extract_json_ld(soup)
    product_jsonld = find_product_jsonld(json_ld_items)

    name = extract_name(soup, product_jsonld)
    brand = infer_brand(name)

    price_info = extract_price(soup, product_jsonld)
    rating_info = extract_rating(soup, product_jsonld)

    variants = extract_variants(soup, name)
    colors = extract_colors(soup)
    specs = extract_specs(soup)
    highlights = extract_highlights(soup)
    description = extract_description(soup)
    comments_reviews = extract_comments_and_reviews(
        soup=soup,
        product_jsonld=product_jsonld,
        include_comments=include_comments,
    )

    ram_storage = extract_ram_storage(name)

    product = {
        "id": make_product_id(url),
        "source": "CellphoneS",
        "category": "Điện thoại",
        "url": normalize_url(url),
        "name": name,
        "brand": brand,
        "ram": ram_storage["ram"],
        "storage": ram_storage["storage"],
        "price": price_info,
        "rating": rating_info,
        "variants": variants,
        "colors": colors,
        "specs": specs,
        "highlights": highlights,
        "description": description,
        "reviews": comments_reviews["reviews"],
        "comments": comments_reviews["comments"],
        "comment_note": comments_reviews["comment_note"],
        "images": extract_images(soup, product_jsonld),
    }

    return product


def product_to_rag_text(product: Dict) -> str:
    name = product.get("name", "")
    brand = product.get("brand", "")
    price = product.get("price", {})
    rating = product.get("rating", {})

    current_price = price.get("current_price") or "chưa có dữ liệu"
    old_price = price.get("old_price") or ""
    discount = price.get("discount") or ""

    rating_value = rating.get("rating_value") or "chưa có dữ liệu"
    review_count = rating.get("review_count")

    lines = []

    lines.append(f"Tên sản phẩm: {name}.")
    lines.append(f"Thương hiệu: {brand or 'chưa xác định'}.")
    lines.append("Danh mục: Điện thoại.")
    lines.append(f"Giá bán hiện tại: {current_price}.")

    if old_price:
        lines.append(f"Giá gốc hoặc giá niêm yết: {old_price}.")

    if discount:
        lines.append(f"Ưu đãi/giảm giá: {discount}.")

    if rating_value != "chưa có dữ liệu":
        if review_count is not None:
            lines.append(f"Điểm đánh giá trung bình: {rating_value}/5 dựa trên {review_count} đánh giá.")
        else:
            lines.append(f"Điểm đánh giá trung bình: {rating_value}/5.")
    else:
        lines.append("Điểm đánh giá: chưa có dữ liệu.")

    if product.get("ram"):
        lines.append(f"Dung lượng RAM theo tên sản phẩm: {product['ram']}.")

    if product.get("storage"):
        lines.append(f"Bộ nhớ trong theo tên sản phẩm: {product['storage']}.")

    variants = product.get("variants") or []
    if variants:
        lines.append("\nCác phiên bản liên quan của sản phẩm:")
        for v in variants:
            detail = f"- {v.get('name', '')}"
            if v.get("ram"):
                detail += f", RAM {v['ram']}"
            if v.get("storage"):
                detail += f", bộ nhớ {v['storage']}"
            if v.get("price"):
                detail += f", giá {v['price']}"
            if v.get("url"):
                detail += f", link {v['url']}"
            lines.append(detail + ".")

    colors = product.get("colors") or []
    if colors:
        lines.append("\nCác màu sắc đang hiển thị:")
        for c in colors:
            detail = f"- {c.get('name', '')}"
            if c.get("price"):
                detail += f", giá {c['price']}"
            lines.append(detail + ".")

    specs = product.get("specs") or {}
    if specs:
        lines.append("\nThông số kỹ thuật:")
        for k, v in specs.items():
            lines.append(f"- {k}: {v}.")

    highlights = product.get("highlights") or []
    if highlights:
        lines.append("\nTính năng nổi bật:")
        for h in highlights:
            lines.append(f"- {h}.")

    description = product.get("description") or ""
    if description:
        lines.append("\nMô tả/đánh giá chi tiết từ trang sản phẩm:")
        lines.append(description)

    reviews = product.get("reviews") or []
    if reviews:
        lines.append("\nĐánh giá người dùng:")
        for r in reviews:
            content = r.get("content", "")
            rating = r.get("rating", "")
            if content:
                if rating:
                    lines.append(f"- Đánh giá {rating}/5: {content}")
                else:
                    lines.append(f"- {content}")

    comments = product.get("comments") or []
    if comments:
        lines.append("\nBình luận/Hỏi đáp thu thập được:")
        for c in comments:
            lines.append(f"- {c.get('content', '')}")

    if product.get("comment_note"):
        lines.append(f"\nGhi chú về bình luận: {product['comment_note']}")

    lines.append(f"\nNguồn dữ liệu: {product.get('url', '')}")

    return "\n".join(lines).strip()


def save_outputs(products: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "json/cellphones_phones.json"
    jsonl_path = output_dir / "json/cellphones_phones.jsonl"
    txt_path = output_dir / "txt/cellphones_phones_rag.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with txt_path.open("w", encoding="utf-8") as f:
        for p in products:
            rag_text = product_to_rag_text(p)
            f.write(rag_text)
            f.write("\n\n")
            f.write("=" * 100)
            f.write("\n\n")

    print(f"[DONE] Saved JSON:  {json_path}")
    print(f"[DONE] Saved JSONL: {jsonl_path}")
    print(f"[DONE] Saved TXT:   {txt_path}")


def crawl(max_products: int, delay: float, include_comments: bool):
    session = make_session()

    print("[INFO] Discovering product URLs from sitemap...")
    product_urls = discover_product_urls_from_sitemap(
        session=session,
        max_products=max_products,
        delay=delay,
    )

    if not product_urls:
        print("[WARN] Sitemap failed or empty. Fallback to category page...")
        product_urls = discover_product_urls_from_category(
            session=session,
            max_products=max_products,
            delay=delay,
        )

    product_urls = product_urls[:max_products]

    print(f"[INFO] Total product URLs: {len(product_urls)}")

    products = []
    failed_urls = []

    for url in tqdm(product_urls, desc="Crawling products"):
        html = fetch(session, url, delay=delay)

        if not html:
            failed_urls.append(url)
            continue

        try:
            product = parse_product_page(
                html=html,
                url=url,
                include_comments=include_comments,
            )

            if not product.get("name"):
                print(f"[WARN] Missing name: {url}")

            products.append(product)

        except Exception as e:
            print(f"[PARSE ERROR] {url} -> {e}")
            failed_urls.append(url)

    save_outputs(products, OUTPUT_DIR)

    if failed_urls:
        fail_path = OUTPUT_DIR / "failed_urls.txt"
        with fail_path.open("w", encoding="utf-8") as f:
            for url in failed_urls:
                f.write(url + "\n")

        print(f"[WARN] Failed URLs saved to: {fail_path}")

    print(f"[SUMMARY] Success: {len(products)} / {len(product_urls)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-products", type=int, default=200)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--include-comments", action="store_true")

    args = parser.parse_args()

    crawl(
        max_products=args.max_products,
        delay=args.delay,
        include_comments=args.include_comments,
    )


if __name__ == "__main__":
    main()
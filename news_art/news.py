from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import re
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


def feed_url(config):
    params = {"hl": config["language"], "gl": config["country"],
              "ceid": f'{config["country"]}:{config["language"].split("-")[0]}'}
    query = config["query"].strip()
    if query:
        params["q"] = query
    return "https://news.google.com/rss" + ("/search" if query else "") + "?" + urlencode(params)


def clean(text):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text or "")).split())


def parse(data, max_age_hours=48, now=None):
    now = now or datetime.now(timezone.utc)
    oldest = now - timedelta(hours=max_age_hours)
    stories, seen = [], set()
    for item in ET.fromstring(data).findall("./channel/item"):
        title = clean(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        source = clean(item.findtext("source")) or "Google News"
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue
        if not title or urlparse(link).scheme not in ("http", "https") or not oldest <= published <= now + timedelta(minutes=10):
            continue
        if title.endswith(" - " + source):
            title = title[:-len(source)-3]
        identifier = hashlib.sha256((item.findtext("guid") or link).encode()).hexdigest()[:24]
        if identifier in seen:
            continue
        seen.add(identifier)
        stories.append({"id": identifier, "headline": title[:1000], "url": link,
                        "source": source[:200], "published": published.isoformat()})
    return stories


def fetch(config):
    request = Request(feed_url(config), headers={"User-Agent": "news-art-pi/0.1 RSS reader"})
    with urlopen(request, timeout=20) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("RSS response exceeded 2 MB")
    stories = parse(data, config["max_age_hours"])
    if not stories:
        raise ValueError("No recent stories matched the feed; keeping the previous poster")
    return stories


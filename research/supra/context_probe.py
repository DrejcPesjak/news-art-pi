"""Laptop-only audit of RSS descriptions and publisher metadata/opening text.

Google's redirect RPC is undocumented and can break. Failures are recorded,
never silently replaced by text from a different article or an invented summary.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from lxml import html
import trafilatura
from news_art.news import clean


def request(url, data=None):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsArtPi research)",
               "Accept-Language": "en-US,en;q=0.9"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded;charset=UTF-8"
    with urlopen(Request(url, headers=headers, data=data), timeout=18) as response:
        body = response.read(4_000_001)
        if len(body) > 4_000_000:
            raise ValueError("Page exceeded 4 MB")
        return body, response.url


def resolve(link):
    body, final = request(link)
    if urlparse(final).hostname != "news.google.com":
        return final
    root = html.fromstring(body)
    elements = root.xpath("//*[@data-n-a-sg]")
    if not elements:
        raise ValueError("Google redirect signature missing")
    element = elements[0]
    context = [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                None, None, None, None, None, 0, 1], "X", "X", 1, [1, 1, 1], 1,
               1, None, 0, 0, None, 0]
    inner = ["garturlreq", context, element.get("data-n-a-id"),
             int(element.get("data-n-a-ts")), element.get("data-n-a-sg")]
    payload = urlencode({"f.req": json.dumps([[["Fbv4je", json.dumps(inner), None, "generic"]]])}).encode()
    body, _ = request("https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je", payload)
    for line in body.decode().splitlines():
        if not line.startswith("[["):
            continue
        for frame in json.loads(line):
            if len(frame) > 2 and frame[:2] == ["wrb.fr", "Fbv4je"] and frame[2]:
                result = json.loads(frame[2])
                if result[0] == "garturlres" and urlparse(result[1]).scheme in ("http", "https"):
                    return result[1]
    raise ValueError("No publisher URL in Google redirect response")


def extract(body, title, url):
    root = html.fromstring(body)
    descriptions = root.xpath('//meta[@property="og:description"]/@content | //meta[@name="description"]/@content')
    summary = next((clean(s) for s in descriptions if len(clean(s)) >= 40 and clean(s).lower().rstrip(".") != title.lower().rstrip(".")), "")
    text = trafilatura.extract(body, url=url, include_comments=False,
                               include_tables=False, favor_precision=True) or ""
    rejected = re.compile(r"accept.*cookies|subscribe to|sign up for|all rights reserved|enable javascript|access denied|verify you are human|thank you for visiting|browser version|compatibility mode|internet explorer|continued support|site without styles", re.I)
    paragraphs = [clean(p) for p in text.splitlines() if len(clean(p)) > 65
                  and clean(p).lower().rstrip(".") not in title.lower()
                  and not rejected.search(p)]
    opening = paragraphs[0] if paragraphs else ""
    # Retain short context only; the experiment is not a full-article archive.
    return summary[:700], opening[:900]


def probe(story, directory):
    path = directory/f'article-{story["feed_index"]:02d}.json'
    if path.exists():
        return json.loads(path.read_text())
    record = dict(story)
    started = time.perf_counter()
    try:
        url = resolve(story["url"])
        record["publisher_url"] = url
        body, final = request(url)
        (directory/f'article-{story["feed_index"]:02d}.html').write_bytes(body)
        record["final_url"] = final
        record["summary"], record["opening"] = extract(body, story["headline"], final)
        record["context_status"] = "available" if record["summary"] or record["opening"] else "no-useful-text"
    except Exception as error:
        record["context_status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record.setdefault("summary", "")
        record.setdefault("opening", "")
    record["fetch_seconds"] = time.perf_counter()-started
    path.write_text(json.dumps(record, indent=2)+"\n")
    print(f'{record["feed_index"]}: {record["context_status"]} / {record.get("publisher_url", "unresolved")}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", type=Path, default=Path("artifacts/context-nlp/google-feed.xml"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 2, 5, 10, 11, 13, 18, 22, 27, 30, 31, 37])
    parser.add_argument("--output", type=Path, default=Path("artifacts/context-nlp"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = ET.fromstring(args.feed.read_bytes()).findall("./channel/item")
    stories, audit = [], []
    for i, item in enumerate(items):
        source = clean(item.findtext("source"))
        title = clean(item.findtext("title"))
        if source and title.endswith(" - "+source):
            title = title[:-len(source)-3]
        description = item.findtext("description") or ""
        related = html.fromstring(description).xpath("//a/text()") if description else []
        audit.append({"index": i, "headline": title, "description_text": clean(description),
                      "related_headline_count": len(related), "has_content_encoded": item.find("{http://purl.org/rss/1.0/modules/content/}encoded") is not None})
        if i in args.indices:
            stories.append({"feed_index": i, "headline": title, "source": source,
                            "url": item.findtext("link"), "rss_description": clean(description),
                            "related_headlines": related[:5]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda story: probe(story, args.output), stories))
    (args.output/"articles.json").write_text(json.dumps(records, indent=2)+"\n")
    (args.output/"feed-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"feed_items": len(items), "tested": len(records),
                      "with_context": sum(r["context_status"] == "available" for r in records)}, indent=2))


if __name__ == "__main__":
    main()

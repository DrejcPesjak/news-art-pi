"""Best-effort publisher context for the selected story, never RSS related titles."""
import json
import re
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from lxml import html
from .news import clean

def request(url, data=None):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsArtPi/1.0)",
               "Accept-Language": "en-US,en;q=0.9"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded;charset=UTF-8"
    with urlopen(Request(url, headers=headers, data=data), timeout=12) as response:
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


REJECTED = re.compile(r"accept.*cookies|subscribe to|sign up for|all rights reserved|enable javascript|access denied|verify you are human|thank you for visiting|browser version|compatibility mode|internet explorer|continued support|site without styles", re.I)


def extract(body, title):
    root = html.fromstring(body)
    descriptions = root.xpath('//meta[@property="og:description"]/@content | //meta[@name="description"]/@content')
    useful = lambda s: len(s) >= 40 and s.lower().rstrip(".") != title.lower().rstrip(".") and not REJECTED.search(s)
    summary = next((clean(s) for s in descriptions if useful(clean(s))), "")
    # Prefer article/main paragraphs. Never use navigation or page headings as context.
    paragraphs = root.xpath('//article//p | //main//p')
    opening = next((clean(p.text_content()) for p in paragraphs if useful(clean(p.text_content())) and len(clean(p.text_content())) >= 65), "")
    return summary[:700], opening[:900]


def context_for(story):
    try:
        url = resolve(story["url"])
        body, final = request(url)
        summary, opening = extract(body, story["headline"])
        return {"status": "available" if summary or opening else "no-useful-text",
                "publisher_url": final, "summary": summary, "opening": opening,
                "context": "\n\n".join(s for s in (summary, opening) if s)}
    except Exception as error:
        return {"status": "failed", "context": "", "error": f"{type(error).__name__}: {error}"}

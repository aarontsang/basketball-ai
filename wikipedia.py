import wikipediaapi


wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="NBAChatbotProject (academic research)"
)

page = wiki.page("National Basketball Association")

seed_pages = [
    "National Basketball Association",
    "History of the NBA",
    "NBA playoffs",
    "List of NBA teams",
    "List of NBA players"
]

def get_linked_pages(page, max_links=200):
    links = []
    for title, link_page in page.links.items():
        if len(links) >= max_links:
            break
        links.append(title)
    return links


KEYWORDS = [
    "NBA", "basketball", "season", "player",
    "team", "playoffs", "draft", "coach"
]

def is_relevant(title):
    title_lower = title.lower()
    return any(k.lower() in title_lower for k in KEYWORDS)



import praw

reddit = praw.Reddit(
    client_id="my client id",
    client_secret="my client secret",
    password="my password",
    user_agent="my user agent",
    username="my username",
)

subreddit = reddit.subreddit("NBA")
print(reddit.read_only)
print(subreddit.display_name)
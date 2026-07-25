"""Deterministic helpers that replace trivial hot-path LLM calls."""

import re


def _norm(text):
  return " ".join(str(text or "").lower().split())


def action_emoji(description):
  text = _norm(description)
  categories = [
    (("sleep", "bed", "rest"), "💤"),
    (("eat", "breakfast", "lunch", "dinner", "food", "cook"), "🍽️"),
    (("talk", "chat", "conversation"), "💬"),
    (("paint", "art", "draw"), "🎨"),
    (("work", "write", "study", "read"), "📚"),
    (("clean", "shower", "bathroom"), "🚿"),
    (("walk", "go", "visit"), "🚶"),
    (("wait",), "⌛"),
  ]
  for words, emoji in categories:
    if any(word in text for word in words):
      return emoji
  return "🙂"


def normalize_action_description(description):
  text = re.sub(r"\s+", " ", str(description or "")).strip()
  return text or "idle"


def action_event_triple(actor, description):
  return (actor, "is", normalize_action_description(description))


def object_event_triple(game_object, description):
  obj = normalize_action_description(game_object)
  return (obj, "is used for", normalize_action_description(description))


def heuristic_poignancy(event_type, description):
  text = _norm(description)
  if "is idle" in text or text == "idle":
    return 1
  if any(word in text for word in ("whisper", "history", "remember", "secret")):
    return 7
  if any(word in text for word in ("promise", "plan", "commit", "invite", "appointment", "meeting", "deadline", "conflict", "argue", "fight", "important")):
    return 7
  if event_type == "chat" or any(word in text for word in ("talk", "chat", "conversation", "social", "friend")):
    return 5
  if any(word in text for word in ("eat", "breakfast", "lunch", "dinner", "work", "study", "clean", "chore", "cook")):
    return 3
  if any(word in text for word in ("sleep", "walk", "routine", "wait")):
    return 2
  if event_type == "thought":
    return 4
  return 2

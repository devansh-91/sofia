"""Decide whether a message is English or Hindi/Hinglish, and tell Sofia which to reply in.

The 4B model can't reliably pick the reply language on its own, so we detect it
here and append an explicit instruction to every user message.
"""
import re

ENGLISH = "english"
HINGLISH = "hinglish"

# Must match the bracketed instructions in models/Sofia.Modelfile's examples.
REPLY_INSTRUCTION = {
    ENGLISH: "[Reply only in casual English.]",
    HINGLISH: "[Reply in simple Roman-script Hinglish.]",
}

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Hinglish borrows English nouns freely ("office", "mood", "boss"), so nouns say
# nothing about the language. Grammar words do: count those on each side. Words
# that exist in both ("to", "me", "main", "do", "the", "is", "in", "he", "hi",
# "are", "or", "so", "par", "mat", "sun", "ho", "us", "ab") are left out of both.
_HINDI_WORDS = frozenset("""
hai hain hoon hun hu nahi nahin kya kyun kyu kyon kaise kaisa kaisi kab kahan kaun ka ki ke ko se mein ne pe
mera meri mere tera teri tere tu tum tumhara tumhari mujhe mujhse tujhe tujhse tumse humein hume hamara
aaj abhi yaar bhi toh tha thi raha rahi rahe kar karo karu karun karna karenge kiya gaya gayi gaye hua hui
aur lekin bahut accha acha achha theek thik haan kuch koi kisi sab bol bolo bata batao chal chalo chalein
wala wali wale apna apni apne yeh ye woh wo kyunki agar phir bas bura neend akela ghar khana paani dost pyaar
gussa dil mann suno suna sunao dekh dekho jao aao kal raat subah ladai gaali gaaliyan bhai didi ek
kitna kitni kitne itna itni zyada thoda thodi sach jhooth pata samajh laga lagi lagta lagti hota hoti hoga hogi
karta karti sakta sakti chahiye liye saath baad pehle kabhi hamesha bilkul zaroor matlab waise vaise
isko usko unko uska uski iska isme usme jab tab jaise idhar udhar yahan wahan andar bahar kaam baat
khush udaas pasand ganda gandi diya di de dena lena milna mil
""".split())

_ENGLISH_WORDS = frozenset("""
i i'm im i've i'll you you're your we they she him her his it it's its this that that's these those
what what's why how how's when where who which was were am be been being have has had having will would
can can't could should shouldn't don't doesn't didn't won't isn't aren't not and but with for from of at on
about a an any some there here then than because if my mine our their them
""".split())


def detect(text: str, previous: str = HINGLISH) -> str:
    """Return ENGLISH or HINGLISH; `previous` wins when the text doesn't say either way."""
    if _DEVANAGARI.search(text):
        return HINGLISH
    words = re.findall(r"[a-z']+", text.lower())
    hindi = sum(w in _HINDI_WORDS for w in words)
    english = sum(w in _ENGLISH_WORDS for w in words)
    if hindi > english:
        return HINGLISH
    if english > hindi:
        return ENGLISH
    # No grammar words at all but a full sentence of Roman text: treat as English.
    if hindi == 0 and len(words) >= 4:
        return ENGLISH
    return previous


def with_reply_instruction(text: str, language: str) -> str:
    return f"{text}\n{REPLY_INSTRUCTION[language]}"

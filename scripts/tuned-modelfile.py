"""Print models/Sofia.Modelfile for the tuned brain: FROM sofia-tuned-base, without the few-shot
examples (the fine-tuning replaces them). scripts/sofia.sh and scripts/finetune.sh build it."""
import re
import sys
from pathlib import Path

text = (Path(__file__).resolve().parent.parent / "models" / "Sofia.Modelfile").read_text()
text = re.sub(r'^MESSAGE (?:user|assistant) """.*?"""\n', "", text, flags=re.S | re.M)
tuned, found = re.subn(r"^FROM sofia-base$", "FROM sofia-tuned-base", text, flags=re.M)
if not found:
    sys.exit("models/Sofia.Modelfile no longer starts FROM sofia-base")
sys.stdout.write(tuned)

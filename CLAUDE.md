# CLAUDE.md

**Read [AGENTS.md](AGENTS.md).** It is the primary instruction file for this repo and follows the
[agents.md](https://agents.md) convention, so every agent working here reads the same thing.

This file exists only because Claude Code reads `CLAUDE.md` first. Keeping the content in
AGENTS.md rather than duplicating it here is deliberate: a rule that only one collaborator's tool
ever sees is worse than no rule at all.

> **reflow2 is installed here.** The design graph is this project's memory. It was recovered by an
> adopt pass over the whole workspace, and it holds open questions that are waiting on a person —
> consult it before writing or changing code.

**The skills are served by the MCP server, not stored in this repo.** They will never appear in a
skills list or as files on disk. Call `list_skills` to see them and `get_skill` to read one in
full, and call `get_instructions` before the first design action of a session.

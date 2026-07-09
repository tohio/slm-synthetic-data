# Documentation Directive

This file defines how documentation is structured in this repository. Follow it
whenever creating or updating a README or doc — whether that's a human
contributor or an LLM assistant.

## 1. Decide where the doc belongs

Ask, in order:

1. **Is this the repo root?** → `README.md` at repo root. Only one of these
   should exist.
2. **Does this content describe one specific code folder** (its purpose,
   contents, internal conventions)? → a `README.md` *inside that folder*
   (sub-README). Only add one if the folder's purpose isn't obvious from its
   name, or it has non-trivial internal structure/conventions worth
   documenting. Don't add a sub-README to every folder by default.
3. **Everything else** (architecture, decisions, guides, process, reference —
   anything not scoped to a single code folder) → lives under `/docs`, and
   must be linked from `docs/README.md` (or a nested index like
   `docs/adr/README.md`). Never float a project-level doc at repo root or
   bolt it onto a code folder it doesn't belong to.

## 2. Use the right skeleton for the doc type

**READMEs describe current state, not history.** Do not add changelog-style
entries, dated "added X / fixed Y" notes, or append-only decision logs to
any README. That kind of content is out of scope for this directive — if a
project wants a changelog, it's tracked separately (e.g. commit history or
release notes), not inside a README skeleton. Decision history specifically
belongs in ADRs (see Section 2D), which are separate from README files for
exactly this reason.

### A. Root `README.md`
Audience: newcomers/evaluators deciding whether to use the project.
Section order:
1. Title + one-line description 
2. Overview (2-4 sentences: what problem this solves, why it exists)
3. Architecture (diagram placeholder + brief component explanation) —
   directly after Overview, before Features
4. Features (bullets)
5. Demo / Screenshot (optional)
6. Getting Started → Prerequisites, Installation, Configuration, Usage
   (copy-pasteable commands only)
7. Project Structure (top-level tree; note sub-READMEs where they exist)
8. Documentation (link to `docs/README.md` — never duplicate its content here)
9. Testing
10. Roadmap / Status (optional)
11. License badge (no build-status badge)
12. Acknowledgments / Contact (optional)

Do not include a Contributing section here.

### B. Sub-README (`<folder>/README.md`)
Audience: developers already in the codebase, oriented on this folder.
Sections:
- Title (folder name)
- Purpose (what this folder owns; what it deliberately does not do)
- Contents (directory tree of this folder)
- Key Files (table — only if some files are non-obvious/load-bearing)
- How It Fits In (relation to the rest of the system; link to the
  architecture doc rather than re-explaining it)
- Usage / API (minimal example, if this is a reusable module)
- Conventions (only where they deviate from repo-wide conventions)
- Gotchas (optional)

### C. `/docs` index (`docs/README.md`, or nested e.g. `docs/adr/README.md`)
Audience: anyone trying to find the right doc.
Default to the **minimal** form:
- Title
- One-line description
- Flat bullet list of links, each with a short inline description, using
  concrete entries (not generic placeholders)

Only upgrade to categorized sections (Architecture & Design / Guides /
Reference / Process, etc.) once the flat list is hard to scan — roughly
7-8+ entries. Add a short note on the docs taxonomy (e.g. "guides are
task-oriented, reference is lookup-oriented, ADRs are a historical record
and shouldn't be edited after acceptance") only if that grouping logic
isn't self-evident from the category names.

### D. Standalone doc inside `/docs` (e.g. `docs/architecture.md`, `docs/guides/deployment.md`)
Audience: whoever the index sent here.
Universal minimum:
- Title
- One-line description of what it covers and who it's for
Beyond that, shape follows content type:
- **Decision record (ADR)**: Status / Context / Decision / Consequences
- **Guide**: task-oriented, step-based
- **Reference**: lookup-oriented (tables, structured sections)
- **Architecture doc**: diagram + component descriptions
Close with a "See Also" section linking related docs where useful.

## 3. Formatting rules

- Use `~~~` (tilde) fences for any code block that itself needs to contain
  fenced code examples, to avoid breaking Markdown rendering.
- Skeletons/templates should contain structural headers and one-line
  guidance notes, not fully worked examples, unless explicitly asked to
  populate real content.
- Keep every doc as short as its purpose allows. Index files in particular
  should stay link lists, not summaries of the content they link to.

## 4. When creating a new doc, check

- [ ] Does this belong at root, in a code folder, or in `/docs`? (Section 1)
- [ ] Is it linked from the relevant index if it's under `/docs`?
- [ ] Does it use the matching skeleton for its type? (Section 2)
- [ ] Is it as short as it can be while still serving its audience?
- [ ] If it duplicates content that lives elsewhere, replace the duplication
      with a link instead.
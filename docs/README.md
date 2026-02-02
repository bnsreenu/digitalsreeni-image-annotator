# Documentation

This directory contains comprehensive documentation for the DigitalSreeni Image Annotator project.

## arc42 Architecture Documentation

The [arc42/](arc42/) folder contains detailed architecture documentation following the arc42 template:

### Quick Navigation

**For Developers:**
- Start here: [Building Block View](arc42/05_building_block_view.md) - Understand components and structure
- See workflows: [Runtime View](arc42/06_runtime_view.md) - How things work at runtime
- Common patterns: [Cross-cutting Concepts](arc42/08_crosscutting_concepts.md) - Coordinate systems, conversions

**For Architects:**
- Design rationale: [Architecture Decisions](arc42/09_architecture_decisions.md) - Why we made key choices
- Constraints: [Architecture Constraints](arc42/02_architecture_constraints.md) - Technical limitations
- Technical debt: [Risks and Technical Debt](arc42/11_risks_and_technical_debt.md) - Known issues

**For Everyone:**
- Project overview: [Introduction and Goals](arc42/01_introduction_and_goals.md)
- Terminology: [Glossary](arc42/12_glossary.md) - Terms, acronyms, data structures

### Full arc42 Documentation Index

| # | Section | File |
|---|---------|------|
| 1 | Introduction and Goals | [01_introduction_and_goals.md](arc42/01_introduction_and_goals.md) |
| 2 | Architecture Constraints | [02_architecture_constraints.md](arc42/02_architecture_constraints.md) |
| 5 | Building Block View | [05_building_block_view.md](arc42/05_building_block_view.md) |
| 6 | Runtime View | [06_runtime_view.md](arc42/06_runtime_view.md) |
| 8 | Cross-cutting Concepts | [08_crosscutting_concepts.md](arc42/08_crosscutting_concepts.md) |
| 9 | Architecture Decisions | [09_architecture_decisions.md](arc42/09_architecture_decisions.md) |
| 11 | Risks and Technical Debt | [11_risks_and_technical_debt.md](arc42/11_risks_and_technical_debt.md) |
| 12 | Glossary | [12_glossary.md](arc42/12_glossary.md) |

See [arc42/README.md](arc42/README.md) for more details on the arc42 template structure.

## Other Documentation

- **[CLAUDE.md](../CLAUDE.md)** - Quick reference guide for Claude Code (in repository root)
- **[README.md](../README.md)** - User-facing documentation (in repository root)

## Contributing to Documentation

When making changes to the codebase:

1. **Update arc42 docs** when architecture changes:
   - Add/update ADRs in [Architecture Decisions](arc42/09_architecture_decisions.md)
   - Update component descriptions in [Building Block View](arc42/05_building_block_view.md)
   - Document new workflows in [Runtime View](arc42/06_runtime_view.md)
   - Add new terms to [Glossary](arc42/12_glossary.md)

2. **Keep CLAUDE.md lean** - Only quick reference info, point to arc42 for details

3. **Document constraints** - Add platform issues, known bugs to [Risks and Technical Debt](arc42/11_risks_and_technical_debt.md)

4. **Explain decisions** - Significant architectural choices deserve ADRs in section 9
